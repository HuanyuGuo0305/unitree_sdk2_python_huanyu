"""
Lightweight MuJoCo debug visualizer for the B2WZ1 hierarchical retrieval
sim2real deployment.

The scene is driven purely by forward kinematics (mj_forward) from REAL
telemetry -- there is no physics stepping, no contacts, no dynamics. It
renders:

    - the live B2W + Z1 kinematic pose (from measured joint positions)
    - the detected object position                  (perception, Base frame)
    - the detected retrieval-target position         (perception, Base frame)
    - the decoded high-level end-effector command    (PLB frame)
    - (optional) the training-defined gripper center, colored by the
      grasp-confidence proxy, to help interpret grasp behavior

The real-time 50-Hz control thread only ever calls push_state(), which
stores plain data under a short lock and returns immediately. All MuJoCo
calls (mj_forward, viewer.sync, ...) happen on the visualizer's own thread,
so the control loop's timing is never affected by rendering.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

import mujoco
import mujoco.viewer

from utils.math import (
    euler_xyz_from_quat_wxyz,
    quat_apply_wxyz,
    quat_from_yaw_wxyz,
    quat_normalize_wxyz,
    quat_unique_wxyz,
)


# Matches B2WZ1HierarchicalRetrievalController.policy_joint_names: leg12 + wheel4.
B2W_POLICY_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
]

Z1_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
Z1_GRIPPER_JOINT_NAME = "jointGripper"
ROOT_FREE_JOINT_NAME = "floating_base_joint"


def _joint_qpos_adr(m: "mujoco.MjModel", joint_name: str) -> int:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"MuJoCo joint not found in visualizer XML: {joint_name!r}")
    return int(m.jnt_qposadr[jid])


def _make_arrow_mat(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix whose local +Z axis aligns with `direction`."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1.0e-8:
        return np.eye(3, dtype=np.float64)

    z_axis = direction / norm
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(z_axis, up_hint)) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up_hint, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1.0e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1.0e-8)

    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)


def _add_sphere(scene, pos, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _absolutize_asset_dirs(xml_text: str, base_dir: str) -> str:
    """
    Rewrite relative <compiler> asset directories to absolute paths.

    Required because the scene-augmented model is compiled from a STRING, and
    MuJoCo resolves relative asset paths against the process CWD in that case
    rather than against the original XML's directory.
    """
    def _sub(match: "re.Match") -> str:
        attr, value = match.group(1), match.group(2)
        if os.path.isabs(value):
            return match.group(0)
        return f'{attr}="{os.path.join(base_dir, value)}"'

    return re.sub(
        r'\b(meshdir|texturedir|assetdir)="([^"]*)"',
        _sub,
        xml_text,
    )


def _scene_xml(
    show_light: bool,
    show_ground: bool,
    ground_z: float,
    ground_half_extent: float,
) -> str:
    """
    MJCF fragment adding a headlight, a directional light and a textured
    ground plane. All names are viz_-prefixed so they cannot collide with the
    robot model's own elements.

    The floor is visual-only (contype/conaffinity 0); the visualizer never
    steps physics, but this keeps it out of any collision computation.
    """
    parts = [
        '<visual>'
        '<headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"'
        ' specular="0 0 0"/>'
        '<rgba haze="0.15 0.25 0.35 1"/>'
        '<global azimuth="90" elevation="-20"/>'
        '</visual>'
    ]

    assets = [
        '<texture type="skybox" builtin="gradient"'
        ' rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>'
    ]

    body = []

    if show_light:
        body.append(
            '<light name="viz_light" pos="0 0 3.0" dir="0 0 -1"'
            ' directional="true"/>'
        )

    if show_ground:
        assets.append(
            '<texture type="2d" name="viz_groundplane" builtin="checker"'
            ' mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"'
            ' markrgb="0.8 0.8 0.8" width="300" height="300"/>'
        )
        assets.append(
            '<material name="viz_groundplane" texture="viz_groundplane"'
            ' texuniform="true" texrepeat="5 5" reflectance="0.2"/>'
        )

        # size "0 0 spacing" means an infinite plane in MJCF.
        half = float(ground_half_extent)
        size = (
            f'{half} {half} 0.05'
            if half > 0.0
            else '0 0 0.05'
        )

        body.append(
            f'<geom name="viz_floor" type="plane" pos="0 0 {float(ground_z)}"'
            f' size="{size}" material="viz_groundplane"'
            ' contype="0" conaffinity="0"/>'
        )

    parts.append('<asset>' + ''.join(assets) + '</asset>')

    if body:
        parts.append('<worldbody>' + ''.join(body) + '</worldbody>')

    return ''.join(parts)


def _load_model(
    xml_path: str,
    show_light: bool,
    show_ground: bool,
    ground_z: float,
    ground_half_extent: float,
) -> "mujoco.MjModel":
    """
    Compile the robot XML with a light/ground scene added.

    Falls back to compiling the untouched file if the augmented string fails
    to compile, so a scene-injection problem can never stop the visualizer
    from showing the robot.
    """
    if not (show_light or show_ground):
        return mujoco.MjModel.from_xml_path(xml_path)

    try:
        base_dir = os.path.dirname(os.path.abspath(xml_path))

        with open(xml_path, "r") as f:
            xml_text = f.read()

        xml_text = _absolutize_asset_dirs(xml_text, base_dir)

        scene = _scene_xml(
            show_light=show_light,
            show_ground=show_ground,
            ground_z=ground_z,
            ground_half_extent=ground_half_extent,
        )

        close = xml_text.rindex("</mujoco>")
        augmented = xml_text[:close] + scene + xml_text[close:]

        return mujoco.MjModel.from_xml_string(augmented)

    except Exception as exc:
        print(
            "[MJ-VIZ][WARN] light/ground injection failed, falling back to "
            "the plain robot model: " + repr(exc)
        )
        return mujoco.MjModel.from_xml_path(xml_path)


def _add_capsule(scene, p0, p1, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1.0e-8:
        return

    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, 0.5 * length, 0.0], dtype=np.float64),
        0.5 * (p0 + p1),
        _make_arrow_mat(diff).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


class MujocoDebugVisualizer:
    """
    Background MuJoCo viewer that renders the current B2WZ1 + Z1 pose plus
    perception/command debug markers. Not a simulator: qpos is written
    directly from real telemetry and only mj_forward (FK) is used.
    """

    def __init__(
        self,
        xml_path: str,
        update_hz: float = 30.0,
        object_marker_radius: float = 0.03,
        retrieval_marker_radius: float = 0.05,
        ee_target_sphere_radius: float = 0.03,
        ee_target_axis_len: float = 0.20,
        ee_target_axis_radius: float = 0.01,
        show_gripper_center: bool = True,
        gripper_center_marker_radius: float = 0.018,
        show_floor: bool = True,
        floor_half_extent: float = 3.0,
        show_light: bool = True,
        ground_z: float = 0.0,
        show_world_origin: bool = False,
        world_origin_axis_len: float = 0.4,
        camera_follow: bool = False,
        camera_follow_smoothing: float = 0.15,
    ):
        # The robot-only MJCF has no light and no floor, so both are injected
        # into the model here (real geometry, not scene decorations).
        self._m = _load_model(
            xml_path=xml_path,
            show_light=show_light,
            show_ground=show_floor,
            ground_z=ground_z,
            ground_half_extent=floor_half_extent,
        )
        self._d = mujoco.MjData(self._m)

        self._update_hz = float(update_hz)

        self._object_marker_radius = float(object_marker_radius)
        self._retrieval_marker_radius = float(retrieval_marker_radius)
        self._ee_target_sphere_radius = float(ee_target_sphere_radius)
        self._ee_target_axis_len = float(ee_target_axis_len)
        self._ee_target_axis_radius = float(ee_target_axis_radius)
        self._show_gripper_center = bool(show_gripper_center)
        self._gripper_center_marker_radius = float(gripper_center_marker_radius)

        # World-frame view options. They only do anything once a caller starts
        # pushing an absolute "base_pos_w"; with a robot-centric feed (the
        # camera/VO deployment) the robot never leaves the origin and there is
        # nothing to follow or to reference.
        self._show_world_origin = bool(show_world_origin)
        self._world_origin_axis_len = float(world_origin_axis_len)
        self._camera_follow = bool(camera_follow)
        self._camera_follow_smoothing = float(
            np.clip(camera_follow_smoothing, 0.0, 1.0)
        )
        self._cam_lookat: Optional[np.ndarray] = None

        self._root_qpos_adr = _joint_qpos_adr(self._m, ROOT_FREE_JOINT_NAME)
        self._b2w_qpos_adr = [
            _joint_qpos_adr(self._m, name) for name in B2W_POLICY_JOINT_NAMES
        ]
        self._arm_qpos_adr = [
            _joint_qpos_adr(self._m, name) for name in Z1_ARM_JOINT_NAMES
        ]
        self._gripper_qpos_adr = _joint_qpos_adr(self._m, Z1_GRIPPER_JOINT_NAME)

        self._lock = threading.Lock()
        self._state: Optional[Dict[str, Any]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Control-thread-facing API. Must stay cheap and non-blocking.
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run,
            name="mj-debug-visualizer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def configure_world_view(
        self,
        show_world_origin: Optional[bool] = None,
        camera_follow: Optional[bool] = None,
        world_origin_axis_len: Optional[float] = None,
        camera_follow_smoothing: Optional[float] = None,
    ) -> None:
        """
        Enable the absolute-world extras after construction.

        Exists so a caller that renders in a real world frame can opt in
        without the construction site having to know about it.
        """
        if show_world_origin is not None:
            self._show_world_origin = bool(show_world_origin)
        if camera_follow is not None:
            self._camera_follow = bool(camera_follow)
        if world_origin_axis_len is not None:
            self._world_origin_axis_len = float(world_origin_axis_len)
        if camera_follow_smoothing is not None:
            self._camera_follow_smoothing = float(
                np.clip(camera_follow_smoothing, 0.0, 1.0)
            )

    def push_state(self, **kwargs: Any) -> None:
        """
        Store the latest telemetry snapshot. Callers must pass already-copied
        arrays/scalars (no references into mutable controller state), since
        the visualizer thread reads this dict asynchronously.
        """
        with self._lock:
            self._state = kwargs

    def update_state(self, **kwargs: Any) -> None:
        """
        Merge extra fields into the snapshot most recently pushed.

        Lets a caller add or override individual keys -- an absolute
        `base_pos_w`, say -- on top of a push it made through a shared code
        path, without having to restate every field. No-op before the first
        push_state(). The same copied-data rule applies.
        """
        with self._lock:
            if self._state is not None:
                self._state.update(kwargs)

    # ------------------------------------------------------------------
    # Visualizer-thread-only from here down.
    # ------------------------------------------------------------------

    def _run(self) -> None:
        with mujoco.viewer.launch_passive(self._m, self._d) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            viewer.cam.lookat[:] = [0.0, 0.0, 0.5]

            period = 1.0 / max(self._update_hz, 1.0)

            while viewer.is_running() and not self._stop_event.is_set():
                loop_start = time.perf_counter()

                with self._lock:
                    state = self._state

                if state is not None:
                    self._apply_state(state)
                    mujoco.mj_forward(self._m, self._d)
                    self._draw_markers(viewer, state)
                    self._update_camera(viewer, state)

                viewer.sync()

                remaining = period - (time.perf_counter() - loop_start)
                if remaining > 0.0:
                    time.sleep(remaining)

    @staticmethod
    def _base_pos_w(state: Dict[str, Any]) -> np.ndarray:
        """
        Absolute base position, when the caller knows one.

        A mocap-driven feed pushes the measured world position. A camera/VO
        feed knows only the height, so the robot is drawn on the world Z axis
        exactly as before -- the whole scene is then robot-relative, which is
        all that feed can honestly support.
        """
        pos = state.get("base_pos_w")
        if pos is not None:
            return np.asarray(pos, dtype=np.float64).reshape(3)
        return np.array([0.0, 0.0, float(state["base_height"])], dtype=np.float64)

    def _apply_state(self, state: Dict[str, Any]) -> None:
        d = self._d

        base_quat_wxyz = np.asarray(state["base_quat_wxyz"], dtype=np.float64).reshape(4)

        d.qpos[self._root_qpos_adr:self._root_qpos_adr + 3] = self._base_pos_w(state)
        d.qpos[self._root_qpos_adr + 3:self._root_qpos_adr + 7] = base_quat_wxyz

        b2w_joint_pos = np.asarray(state["b2w_joint_pos_policy"], dtype=np.float64).reshape(16)
        for adr, q in zip(self._b2w_qpos_adr, b2w_joint_pos):
            d.qpos[adr] = q

        z1_q = np.asarray(state["z1_q"], dtype=np.float64).reshape(6)
        for adr, q in zip(self._arm_qpos_adr, z1_q):
            d.qpos[adr] = q

        d.qpos[self._gripper_qpos_adr] = float(state["gripper_q_training"])

    def _draw_world_origin(self, scene, ground_z: float) -> None:
        """RGB triad at the mocap world origin, as a fixed spatial reference."""
        origin = np.array([0.0, 0.0, ground_z], dtype=np.float32)
        length = self._world_origin_axis_len
        for axis, rgba in enumerate(
            ([1.0, 0.3, 0.3, 0.75], [0.3, 1.0, 0.3, 0.75], [0.4, 0.5, 1.0, 0.75])
        ):
            end = origin.copy()
            end[axis] += length
            _add_capsule(scene, origin, end, 0.006, rgba)

    def _update_camera(self, viewer, state: Dict[str, Any]) -> None:
        """
        Keep the robot in frame as it drives around the capture volume.

        Only the look-at point is driven, so orbiting and zooming with the
        mouse keep working; it is smoothed so the view does not jitter at the
        mocap's noise level.
        """
        if not self._camera_follow or state.get("base_pos_w") is None:
            return

        target = self._base_pos_w(state)
        if self._cam_lookat is None:
            self._cam_lookat = target.copy()
        else:
            alpha = self._camera_follow_smoothing
            self._cam_lookat += alpha * (target - self._cam_lookat)

        viewer.cam.lookat[:] = self._cam_lookat

    def _draw_markers(self, viewer, state: Dict[str, Any]) -> None:
        viewer.user_scn.ngeom = 0

        base_quat_wxyz = np.asarray(state["base_quat_wxyz"], dtype=np.float32).reshape(4)
        ground_z = float(state.get("ground_z", 0.0))

        # The ground is a real model geom (see _load_model), so nothing is
        # drawn for it here.
        base_pos_w = self._base_pos_w(state).astype(np.float32)

        _, _, base_yaw = euler_xyz_from_quat_wxyz(base_quat_wxyz)
        plb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(quat_from_yaw_wxyz(base_yaw)))
        plb_pos_w = np.array(
            [base_pos_w[0], base_pos_w[1], ground_z], dtype=np.float32
        )

        object_pos_base = state.get("object_pos_base")
        if object_pos_base is not None:
            object_pos_w = base_pos_w + quat_apply_wxyz(
                base_quat_wxyz, np.asarray(object_pos_base, dtype=np.float32)
            )
            _add_sphere(
                viewer.user_scn,
                object_pos_w,
                self._object_marker_radius,
                [0.1, 1.0, 0.1, 0.9],
            )

        retrieval_pos_base = state.get("retrieval_pos_base")
        if retrieval_pos_base is not None:
            retrieval_pos_w = base_pos_w + quat_apply_wxyz(
                base_quat_wxyz, np.asarray(retrieval_pos_base, dtype=np.float32)
            )
            _add_sphere(
                viewer.user_scn,
                retrieval_pos_w,
                self._retrieval_marker_radius,
                [1.0, 0.0, 1.0, 0.9],
            )

        ee_cmd_plb = state.get("ee_cmd_plb")
        if ee_cmd_plb is not None:
            kps_plb = np.asarray(ee_cmd_plb, dtype=np.float32).reshape(3, 3)
            kps_w = np.stack(
                [plb_pos_w + quat_apply_wxyz(plb_quat_w, p) for p in kps_plb],
                axis=0,
            )
            kp0, kp1, kp2 = kps_w
            x_dir = kp1 - kp0
            z_dir = kp2 - kp0
            x_norm = float(np.linalg.norm(x_dir))
            z_norm = float(np.linalg.norm(z_dir))

            _add_sphere(
                viewer.user_scn,
                kp0,
                self._ee_target_sphere_radius,
                [1.0, 0.0, 0.0, 0.9],
            )

            if x_norm > 1.0e-8:
                x_end = kp0 + self._ee_target_axis_len * x_dir / x_norm
                _add_capsule(
                    viewer.user_scn, kp0, x_end,
                    self._ee_target_axis_radius, [1.0, 0.0, 0.0, 0.9],
                )

            if z_norm > 1.0e-8:
                z_end = kp0 + self._ee_target_axis_len * z_dir / z_norm
                _add_capsule(
                    viewer.user_scn, kp0, z_end,
                    self._ee_target_axis_radius, [0.0, 0.0, 1.0, 0.9],
                )

        if self._show_world_origin:
            self._draw_world_origin(viewer.user_scn, ground_z)

        # Raw mocap rigid-body pivots, when a mocap feed supplies them. Seeing
        # the B2 pivot sitting a fixed distance off the rendered root is the
        # quickest visual check that the calibrated offset is still right.
        for marker in state.get("mocap_points") or ():
            pos = marker.get("pos")
            if pos is None:
                continue
            _add_sphere(
                viewer.user_scn,
                np.asarray(pos, dtype=np.float32),
                float(marker.get("radius", 0.02)),
                marker.get("rgba", [1.0, 1.0, 1.0, 0.8]),
            )

        if self._show_gripper_center:
            gripper_center_pos_base = state.get("gripper_center_pos_base")
            if gripper_center_pos_base is not None:
                gripper_center_pos_w = base_pos_w + quat_apply_wxyz(
                    base_quat_wxyz,
                    np.asarray(gripper_center_pos_base, dtype=np.float32),
                )
                grasp_active = bool(state.get("grasp_confidence_proxy", False))
                rgba = [0.0, 1.0, 0.0, 0.95] if grasp_active else [1.0, 1.0, 0.0, 0.95]
                _add_sphere(
                    viewer.user_scn,
                    gripper_center_pos_w,
                    self._gripper_center_marker_radius,
                    rgba,
                )
