"""
Interactive MuJoCo tool for calibrating the fixed offset between the OptiTrack
rigid body glued to the B2W and the robot's true `base_link` (root) frame.

Motive tracks a marker cluster, and the rigid-body frame it streams sits
wherever Motive happened to place its pivot -- close to the robot body, but
not at the root frame the policies and MuJoCo model use. This tool solves for
the constant transform between the two:

    p_root^W = p_mocap^W + R(q_mocap^W) . p_offset
    q_root^W = q_mocap^W (x) q_offset

`(p_offset, q_offset)` is expressed in the mocap rigid-body frame, so it is a
fixed mechanical attachment and does not change as the robot moves.

The B2W is drawn in MuJoCo at that reconstructed root pose using LIVE joint
angles from the robot's DDS LowState, together with the raw mocap markers and
both frame triads, so a wrong offset is immediately visible: the model floats
above or sinks into the floor, leans relative to the measured IMU gravity, or
sits off to the side of its own markers.

Three of the six DoF can be solved automatically:

    roll/pitch  the B2 IMU knows which way is down; mocap knows which way
                world -Z is. The offset rotation that reconciles them is
                solved in closed form.
    z           on flat ground with all four wheels down, the lowest wheel
                surface point must touch `ground_z`.

The remaining three (x, y, yaw) have no absolute reference from a standing
robot: yaw can be recovered by driving the robot in a straight line (the base
+X axis must line up with the travel direction), and x/y are nudged by hand
against the rendered marker cloud.

Run (from the repository root):

    python3 deploy/b2w_mocap_root_calibration.py \
        deploy/configs/b2w_mocap_root_calibration.yaml

Keys are read from the TERMINAL, not the MuJoCo window -- MuJoCo's own viewer
already binds nearly every letter to a rendering flag. Keep the terminal
focused and the viewer visible; press `?` for the key map.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import queue
import select
import shutil
import termios
import threading
import time
import tty
from typing import List, Optional, Tuple

import numpy as np
import yaml

import mujoco
import mujoco.viewer

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo

from utils.natnet_client import NatNetClient

# Reused verbatim from the deployment visualizer: b2w.xml is a robot-only
# model with no light and no floor, and these inject both into the compiled
# model (with an asset-path fixup, since it is compiled from a string).
from utils.mj_visualizer import _add_capsule, _add_sphere, _load_model


POLICY_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
]

HARDWARE_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
]

ROOT_FREE_JOINT_NAME = "floating_base_joint"
WHEEL_BODY_NAMES = [
    "FL_wheel_link", "FR_wheel_link", "RL_wheel_link", "RR_wheel_link",
]

WORLD_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


from utils.mocap_frames import (  # noqa: F401 - re-exported for tooling
    q_normalize,
    q_conj,
    q_mul,
    q_to_mat,
    q_apply,
    q_apply_inv,
    q_from_axis_angle,
    q_from_rpy,
    rpy_from_q,
    q_log,
    q_exp,
    q_between_vectors,
    wrap_pi,
    RootOffset,
    load_root_offset,
)


class Sample:
    """One frozen (mocap, IMU, joint) triple used by the batch solver."""

    __slots__ = ("p_mocap", "q_mocap", "gravity_base", "joint_pos")

    def __init__(self, p_mocap, q_mocap, gravity_base, joint_pos) -> None:
        self.p_mocap = np.asarray(p_mocap, dtype=np.float64).reshape(3).copy()
        self.q_mocap = q_normalize(q_mocap)
        self.gravity_base = np.asarray(gravity_base, dtype=np.float64).reshape(3).copy()
        self.joint_pos = np.asarray(joint_pos, dtype=np.float64).reshape(16).copy()


# ======================================================================
# Robot telemetry
# ======================================================================


class B2WLowStateReader:
    """
    DDS LowState subscriber, reduced to what the calibration needs: joint
    positions in policy order and the IMU orientation quaternion.
    """

    def __init__(self, fallback_joint_pos: np.ndarray) -> None:
        self._lock = threading.Lock()
        self._joint_pos = np.asarray(fallback_joint_pos, dtype=np.float64).reshape(16).copy()
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._recv_time = 0.0
        self._msg_count = 0

        self._hw_index_for_policy = [
            HARDWARE_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES
        ]

        self._subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self._subscriber.Init(self._handler, 10)

    def _handler(self, msg: LowStateGo) -> None:
        joint_pos = np.empty(16, dtype=np.float64)
        for policy_idx, hw_idx in enumerate(self._hw_index_for_policy):
            joint_pos[policy_idx] = msg.motor_state[hw_idx].q

        quat = np.array(
            [
                msg.imu_state.quaternion[0],
                msg.imu_state.quaternion[1],
                msg.imu_state.quaternion[2],
                msg.imu_state.quaternion[3],
            ],
            dtype=np.float64,
        )

        with self._lock:
            self._joint_pos = joint_pos
            self._quat = q_normalize(quat)
            self._recv_time = time.monotonic()
            self._msg_count += 1

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        with self._lock:
            age = (
                time.monotonic() - self._recv_time
                if self._recv_time > 0.0
                else float("inf")
            )
            return self._joint_pos.copy(), self._quat.copy(), age, self._msg_count

    def wait_for_first(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                if self._msg_count > 0:
                    return True
            time.sleep(0.02)
        return False


def gravity_in_base(imu_quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Unit gravity direction expressed in the base frame.

    The IMU's world frame is gravity-aligned but yaw-arbitrary, so only this
    direction is usable for calibration -- which is exactly why the gravity
    solver recovers roll/pitch and leaves yaw alone.
    """
    return q_apply_inv(imu_quat_wxyz, WORLD_DOWN)


# ======================================================================
# MuJoCo scene
# ======================================================================


class B2WScene:
    """B2W model held at an externally supplied root pose (FK only, no physics)."""

    def __init__(self, xml_path: str, ground_z: float, floor_half_extent: float) -> None:
        self.model = _load_model(
            xml_path=xml_path,
            show_light=True,
            show_ground=True,
            ground_z=float(ground_z),
            ground_half_extent=float(floor_half_extent),
        )
        self.data = mujoco.MjData(self.model)
        self.ground_z = float(ground_z)

        self._root_adr = self._qpos_adr(ROOT_FREE_JOINT_NAME)
        self._joint_adr = [self._qpos_adr(name) for name in POLICY_JOINT_NAMES]
        self._wheel_geoms = self._collect_wheel_geoms()

    def _qpos_adr(self, joint_name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"joint {joint_name!r} not found in {self.model}")
        return int(self.model.jnt_qposadr[jid])

    def _collect_wheel_geoms(self) -> List[Tuple[int, Optional[np.ndarray]]]:
        """
        Geoms belonging to the four wheel bodies, with their mesh vertices
        cached so the exact lowest contact point can be evaluated later.

        Mesh vertices give the true tyre surface; `geom_rbound` (the fallback
        for primitive geoms) is a bounding sphere and would over-estimate the
        wheel radius by the wheel's half width.
        """
        geoms: List[Tuple[int, Optional[np.ndarray]]] = []
        for body_name in WHEEL_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                raise ValueError(f"body {body_name!r} not found in the MuJoCo model")
            for gid in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gid]) != bid:
                    continue
                verts = None
                if int(self.model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                    mesh_id = int(self.model.geom_dataid[gid])
                    if mesh_id >= 0:
                        start = int(self.model.mesh_vertadr[mesh_id])
                        count = int(self.model.mesh_vertnum[mesh_id])
                        verts = np.asarray(
                            self.model.mesh_vert[start:start + count], dtype=np.float64
                        ).reshape(count, 3)
                geoms.append((gid, verts))

        if not geoms:
            raise ValueError("no wheel geoms found; check WHEEL_BODY_NAMES")
        return geoms

    def set_pose(self, p_root: np.ndarray, q_root: np.ndarray, joint_pos: np.ndarray) -> None:
        qpos = self.data.qpos
        qpos[self._root_adr:self._root_adr + 3] = np.asarray(p_root, dtype=np.float64)
        qpos[self._root_adr + 3:self._root_adr + 7] = q_normalize(q_root)
        for adr, q in zip(self._joint_adr, np.asarray(joint_pos, dtype=np.float64).reshape(16)):
            qpos[adr] = q
        mujoco.mj_forward(self.model, self.data)

    def lowest_wheel_z(self) -> float:
        lowest = float("inf")
        for gid, verts in self._wheel_geoms:
            pos = np.asarray(self.data.geom_xpos[gid], dtype=np.float64)
            if verts is None:
                lowest = min(lowest, float(pos[2] - self.model.geom_rbound[gid]))
                continue
            mat = np.asarray(self.data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            lowest = min(lowest, float(np.min(verts @ mat[2, :]) + pos[2]))
        return lowest


# ======================================================================
# Terminal key input
# ======================================================================


class TerminalKeyReader:
    """
    Single-keypress reader on stdin.

    Keys deliberately do NOT come from the MuJoCo window: mjVISSTRING /
    mjRNDSTRING already claim nearly every letter as a rendering-flag
    shortcut, so a viewer-side binding would silently toggle wireframe or
    transparency while nudging the offset.

    cbreak mode leaves ISIG enabled, so Ctrl+C still interrupts normally.
    """

    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fd = sys.stdin.fileno() if self.enabled else -1
        self._saved: Optional[list] = None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(
            target=self._run, name="calib-keys", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.2)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                chunk = os.read(self._fd, 16)
            except OSError:
                return
            if not chunk:
                return
            self._push(chunk.decode("utf-8", "ignore"))

    def _push(self, text: str) -> None:
        """
        Queue keystrokes, dropping ANSI escape sequences.

        Arrow keys and friends arrive as ESC '[' 'A' inside a single read, so
        a bare ESC is only a real Escape when nothing follows it in the same
        chunk -- otherwise every arrow press would read as "quit".
        """
        i = 0
        while i < len(text):
            char = text[i]
            if char != "\x1b":
                self._queue.put(char)
                i += 1
                continue
            if i == len(text) - 1:
                self._queue.put(char)  # a real Escape
                return
            i += 1
            if i < len(text) and text[i] in "[O":
                i += 1
                while i < len(text) and not ("@" <= text[i] <= "~"):
                    i += 1
            i += 1

    def drain(self) -> List[str]:
        keys: List[str] = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                return keys


# ======================================================================
# Calibrator
# ======================================================================


HELP_TEXT = """
  ---------------------------------------------------------------------
   B2W mocap -> root offset calibration            (keys read from HERE)
  ---------------------------------------------------------------------
   translate   w / s   +x / -x        step is shown in the status line
               a / d   +y / -y
               e / q   +z / -z
   rotate      u / o   roll  + / -
               i / k   pitch + / -
               j / l   yaw   + / -
   step size   n / m   x0.5 / x2

   f   toggle translation frame: mocap-body-local  <->  mocap world
   g   solve roll/pitch NOW from the IMU gravity vector
   b   solve z NOW by dropping the lowest wheel point onto ground_z
   c   capture a sample (re-pose the robot between captures)
   v   batch-solve roll/pitch + vertical offset from ALL samples
   x   clear captured samples
   y   start/stop straight-drive recording, then solve yaw from it
   r   reset offset to the config value
   z   zero the offset
   p   print the offset in full
   t   cycle marker/triad rendering
   ?   this help
 Enter  save the offset to the output YAML
   Esc  quit without saving
  ---------------------------------------------------------------------
   While the robot stands level, only roll, pitch and one position
   component are observable. Solve yaw with 'y' (straight drive); nudge
   x / y by hand against the rendered marker cloud. Capturing samples
   with the robot TILTED makes more of the position offset observable.
  ---------------------------------------------------------------------
"""


class MocapRootCalibrator:
    def __init__(self, cfg: dict, cfg_path: str, use_dds: bool = True) -> None:
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.ground_z = float(cfg.get("ground_z", 0.0))

        self.fallback_joint_pos = np.asarray(
            cfg.get(
                "fallback_joint_pos_policy",
                [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0,
                 -1.5, -1.5, -1.5, -1.5, 0.0, 0.0, 0.0, 0.0],
            ),
            dtype=np.float64,
        ).reshape(16)

        self.scene = B2WScene(
            xml_path=cfg["mujoco_xml_path"],
            ground_z=self.ground_z,
            floor_half_extent=float(cfg.get("floor_half_extent", 4.0)),
        )

        self.initial_offset = RootOffset.from_rpy(
            cfg.get("offset_pos", [0.0, 0.0, 0.0]),
            cfg.get("offset_rpy", [0.0, 0.0, 0.0]),
        )
        self.offset = self.initial_offset.copy()

        self.mocap = NatNetClient(
            server_ip=cfg.get("mocap_server_ip") or None,
            local_ip=str(cfg.get("mocap_local_ip", "")),
            multicast_group=str(cfg.get("mocap_multicast_group", "239.255.42.99")),
            data_port=int(cfg.get("mocap_data_port", 1511)),
            command_port=int(cfg.get("mocap_command_port", 1510)),
            join_multicast=bool(cfg.get("mocap_join_multicast", True)),
            up_axis=str(cfg.get("mocap_up_axis", "auto")),
        )
        self.rigid_body_id = int(cfg.get("mocap_rigid_body_id", 1))
        self.rigid_body_name = cfg.get("mocap_rigid_body_name") or None
        self.mocap_max_age_s = float(cfg.get("mocap_max_age_s", 0.5))

        self.robot: Optional[B2WLowStateReader] = None
        self.use_dds = bool(use_dds)

        self.keys = TerminalKeyReader()

        self.step_pos = float(cfg.get("initial_pos_step_m", 0.01))
        self.step_rot = float(cfg.get("initial_rot_step_deg", 1.0)) * np.pi / 180.0
        self.translate_world_frame = False
        self.render_mode = 2  # 0 none, 1 triads, 2 triads + markers

        self.samples: List[Sample] = []
        self.drive_samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.drive_recording = False

        self.output_path = str(
            cfg.get("output_path", "deploy/configs/b2w_mocap_root_offset.yaml")
        )

        # Latest values shared between the update step and the status line.
        self._last_p_mocap = np.zeros(3, dtype=np.float64)
        self._last_q_mocap = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._mocap_ok = False
        self._mocap_age = float("inf")
        self._tracking_valid = False
        self._gravity_base = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._joint_pos = self.fallback_joint_pos.copy()
        self._imu_age = float("inf")
        self._lowest_z = float("nan")
        self._gravity_residual_deg = float("nan")
        self._quit = False
        self._message = ""

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self.use_dds:
            net = str(self.cfg.get("robot_net_interface", ""))
            print(f"[CALIB] initializing DDS on {net!r} ...")
            ChannelFactoryInitialize(0, net)
            self.robot = B2WLowStateReader(self.fallback_joint_pos)
            if self.robot.wait_for_first(timeout_s=float(self.cfg.get("dds_timeout_s", 10.0))):
                print("[CALIB] LowState stream is live.")
            else:
                print(
                    "[CALIB][WARN] no LowState received. Joint angles fall back to "
                    "the config pose and the gravity solver is unavailable."
                )
        else:
            print("[CALIB] --no-dds: rendering the fallback pose, no IMU solver.")

        self.mocap.start()
        if self.rigid_body_name:
            resolved = self.mocap.rigid_body_id_for_name(self.rigid_body_name)
            if resolved is not None:
                print(
                    f"[CALIB] rigid body {self.rigid_body_name!r} -> streaming id {resolved}"
                )
                self.rigid_body_id = resolved
            else:
                print(
                    f"[CALIB][WARN] rigid body {self.rigid_body_name!r} not in the model "
                    f"definitions; falling back to id {self.rigid_body_id}"
                )

        if self.mocap.wait_for_first_frame(timeout_s=float(self.cfg.get("mocap_timeout_s", 10.0))):
            ids = self.mocap.rigid_body_ids()
            print(f"[CALIB] mocap frames arriving. Rigid body ids in stream: {ids}")
            if self.rigid_body_id not in ids:
                print(
                    f"[CALIB][WARN] configured mocap_rigid_body_id={self.rigid_body_id} "
                    f"is not being streamed."
                )
        else:
            print(
                "[CALIB][WARN] no mocap frames. Check Motive's streaming pane "
                "(broadcast frame data on, correct interface) and the local IP."
            )

    def close(self) -> None:
        self.keys.stop()
        self.mocap.stop()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update_telemetry(self) -> None:
        rb = self.mocap.latest_rigid_body(self.rigid_body_id)
        if rb is not None:
            self._mocap_age = rb.age_s()
            self._tracking_valid = rb.tracking_valid
            fresh = self._mocap_age <= self.mocap_max_age_s
            if fresh and rb.tracking_valid:
                self._last_p_mocap = rb.pos
                self._last_q_mocap = rb.quat_wxyz
                self._mocap_ok = True
            else:
                self._mocap_ok = False
        else:
            self._mocap_ok = False
            self._mocap_age = float("inf")
            self._tracking_valid = False

        if self.robot is not None:
            joint_pos, imu_quat, imu_age, count = self.robot.snapshot()
            self._joint_pos = joint_pos
            self._imu_age = imu_age if count > 0 else float("inf")
            self._gravity_base = gravity_in_base(imu_quat)

    def refresh_scene(self) -> None:
        p_root, q_root = self.offset.apply(self._last_p_mocap, self._last_q_mocap)
        self.scene.set_pose(p_root, q_root, self._joint_pos)
        self._lowest_z = self.scene.lowest_wheel_z()

        if np.isfinite(self._imu_age):
            gravity_world = q_apply(q_root, self._gravity_base)
            cos_err = float(np.clip(np.dot(gravity_world, WORLD_DOWN), -1.0, 1.0))
            self._gravity_residual_deg = float(np.degrees(np.arccos(cos_err)))
        else:
            self._gravity_residual_deg = float("nan")

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _gravity_delta(self, q_mocap: np.ndarray, gravity_base: np.ndarray) -> np.ndarray:
        """
        Mocap-frame rotation correction that makes the IMU's "down" agree with
        the mocap world's "down".

        Both vectors are expressed in the mocap rigid-body frame, so the
        shortest-arc rotation between them touches roll and pitch only and
        leaves the (unobservable) yaw untouched.
        """
        gravity_via_offset = q_apply(self.offset.quat, gravity_base)
        gravity_via_mocap = q_apply_inv(q_mocap, WORLD_DOWN)
        return q_between_vectors(gravity_via_offset, gravity_via_mocap)

    def _ground_constraint(self, q_mocap: np.ndarray, lowest_z: float) -> Tuple[np.ndarray, float]:
        """
        The floor contact condition as one linear constraint on the offset.

        Moving the offset by `delta` (mocap frame) raises the lowest wheel
        point by `normal . delta` in world Z, so lowest_z == ground_z becomes

            normal . delta = residual

        with `normal` the world up-axis expressed in the mocap frame. That
        direction only swings when the robot's ROLL/PITCH changes, which is
        why level-ground samples can never pin more than one component.
        """
        normal = q_apply_inv(q_mocap, np.array([0.0, 0.0, 1.0]))
        return normal, float(self.ground_z - lowest_z)

    def _ground_delta_pos(self, q_mocap: np.ndarray, lowest_z: float) -> np.ndarray:
        """Mocap-frame translation that lands the lowest wheel point on the floor."""
        normal, residual = self._ground_constraint(q_mocap, lowest_z)
        return residual * normal

    def solve_gravity_now(self) -> None:
        if not self._require_mocap() or not self._require_imu():
            return
        before = self._gravity_residual_deg
        self.offset.rotate_local(
            self._gravity_delta(self._last_q_mocap, self._gravity_base)
        )
        self.refresh_scene()
        self._note(
            f"gravity solve: roll/pitch residual {before:.3f} deg -> "
            f"{self._gravity_residual_deg:.3f} deg"
        )

    def solve_ground_now(self) -> None:
        if not self._require_mocap():
            return
        self.refresh_scene()
        before_mm = 1000.0 * (self._lowest_z - self.ground_z)
        self.offset.translate_local(
            self._ground_delta_pos(self._last_q_mocap, self._lowest_z)
        )
        self.refresh_scene()
        self._note(
            f"ground solve: lowest wheel point {before_mm:+.1f} mm -> "
            f"{1000.0 * (self._lowest_z - self.ground_z):+.1f} mm"
        )

    def capture_sample(self) -> None:
        if not self._require_mocap() or not self._require_imu():
            return
        self.samples.append(
            Sample(
                p_mocap=self._last_p_mocap,
                q_mocap=self._last_q_mocap,
                gravity_base=self._gravity_base,
                joint_pos=self._joint_pos,
            )
        )
        self._note(f"captured sample {len(self.samples)}")

    def solve_from_samples(self, iterations: int = 12) -> None:
        """
        Alternating least-squares over every captured sample.

        Each iteration averages the per-sample gravity correction (in rotation-
        vector space) and then the per-sample ground correction, so noise in
        any single mocap frame or IMU reading is averaged out.

        What this can and cannot observe:

            roll/pitch      always, from gravity.
            yaw             never -- the IMU world frame is yaw-arbitrary.
                            Use the straight-drive solver ('y') for yaw.
            position        only ONE component: the ground constraint pins
                            the offset along R(q_offset) . z_world, and while
                            the robot stands level that direction is identical
                            in every sample no matter how the robot is yawed
                            or how tall it stands.

        To make more of the position offset observable, capture samples with
        the robot actually TILTED (two wheels up on a plank, say). Only a
        change in roll/pitch swings the constraint direction within the mocap
        frame; on level ground x/y stay a manual, marker-cloud judgement.
        """
        if not self.samples:
            self._note("no samples captured (press 'c' first)")
            return

        for _ in range(int(iterations)):
            rotvecs = [
                q_log(self._gravity_delta(s.q_mocap, s.gravity_base))
                for s in self.samples
            ]
            self.offset.rotate_local(q_exp(np.mean(np.stack(rotvecs, axis=0), axis=0)))

            normals, residuals = [], []
            for s in self.samples:
                p_root, q_root = self.offset.apply(s.p_mocap, s.q_mocap)
                self.scene.set_pose(p_root, q_root, s.joint_pos)
                normal, residual = self._ground_constraint(
                    s.q_mocap, self.scene.lowest_wheel_z()
                )
                normals.append(normal)
                residuals.append(residual)

            # Least squares rather than an average of the per-sample steps:
            # when the samples span several tilts the constraints disagree and
            # only a joint solve satisfies them all, and lstsq's minimum-norm
            # solution leaves genuinely unobservable directions untouched
            # instead of drifting them.
            delta, _, _, _ = np.linalg.lstsq(
                np.stack(normals, axis=0), np.asarray(residuals), rcond=1.0e-6
            )
            self.offset.translate_local(delta)

        grav_res, ground_res = self._sample_residuals()
        self.refresh_scene()
        self._note(
            f"batch solve over {len(self.samples)} samples: "
            f"gravity rms {grav_res:.3f} deg, ground rms {ground_res:.1f} mm"
        )

    def _sample_residuals(self) -> Tuple[float, float]:
        grav, ground = [], []
        for s in self.samples:
            p_root, q_root = self.offset.apply(s.p_mocap, s.q_mocap)
            gravity_world = q_apply(q_root, s.gravity_base)
            grav.append(
                np.degrees(
                    np.arccos(float(np.clip(np.dot(gravity_world, WORLD_DOWN), -1.0, 1.0)))
                )
            )
            self.scene.set_pose(p_root, q_root, s.joint_pos)
            ground.append(1000.0 * (self.scene.lowest_wheel_z() - self.ground_z))
        return (
            float(np.sqrt(np.mean(np.square(grav)))),
            float(np.sqrt(np.mean(np.square(ground)))),
        )

    def toggle_drive_recording(self) -> None:
        if not self.drive_recording:
            self.drive_samples = []
            self.drive_recording = True
            self._note(
                "yaw recording STARTED - drive the robot straight forward "
                "(>= 1 m), then press 'y' again"
            )
            return

        self.drive_recording = False
        self._solve_yaw_from_drive()

    def _solve_yaw_from_drive(self) -> None:
        """
        Recover the offset yaw by requiring the base +X axis to point along a
        straight-line drive.

        Assumes the robot drove forward (not backward) and held a roughly
        constant heading, which is what makes the travel direction a proxy for
        the body forward axis.
        """
        if len(self.drive_samples) < 10:
            self._note(f"only {len(self.drive_samples)} drive samples; discarded")
            return

        positions = np.stack([p for p, _ in self.drive_samples], axis=0)
        travel = positions[-1, :2] - positions[0, :2]
        distance = float(np.linalg.norm(travel))
        if distance < float(self.cfg.get("yaw_solve_min_distance_m", 0.5)):
            self._note(f"drive was only {distance:.2f} m; need a longer straight run")
            return
        travel_heading = float(np.arctan2(travel[1], travel[0]))

        errors, quats = [], []
        for _, q_mocap in self.drive_samples:
            q_root = q_mul(q_mocap, self.offset.quat)
            forward = q_apply(q_root, [1.0, 0.0, 0.0])
            if float(np.linalg.norm(forward[:2])) < 1.0e-6:
                continue
            errors.append(
                wrap_pi(travel_heading - float(np.arctan2(forward[1], forward[0])))
            )
            quats.append(q_mocap)

        if not errors:
            self._note("could not resolve a forward axis from the drive")
            return

        # Circular mean, so a run straddling +/-pi averages correctly.
        mean_error = float(
            np.arctan2(np.mean(np.sin(errors)), np.mean(np.cos(errors)))
        )
        spread_deg = float(
            np.degrees(
                np.sqrt(np.mean(np.square([wrap_pi(e - mean_error) for e in errors])))
            )
        )

        # The correction is a world-Z rotation, mapped into each sample's
        # mocap frame before being averaged.
        correction = q_from_axis_angle([0.0, 0.0, 1.0], mean_error)
        rotvecs = [
            q_log(q_mul(q_conj(q_mocap), q_mul(correction, q_mocap)))
            for q_mocap in quats
        ]
        self.offset.rotate_local(q_exp(np.mean(np.stack(rotvecs, axis=0), axis=0)))
        self.refresh_scene()
        self._note(
            f"yaw solve over {len(errors)} samples / {distance:.2f} m: "
            f"applied {np.degrees(mean_error):+.2f} deg (heading spread {spread_deg:.2f} deg)"
        )

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _require_mocap(self) -> bool:
        if not self._mocap_ok:
            self._note("no valid mocap pose right now")
            return False
        return True

    def _require_imu(self) -> bool:
        if self.robot is None or not np.isfinite(self._imu_age):
            self._note("no IMU data (needs a live LowState stream)")
            return False
        return True

    def _note(self, message: str) -> None:
        self._message = message
        sys.stdout.write("\r\033[K[CALIB] " + message + "\n")
        sys.stdout.flush()

    def handle_key(self, key: str) -> None:
        translate = {
            "w": (0, +1.0), "s": (0, -1.0),
            "a": (1, +1.0), "d": (1, -1.0),
            "e": (2, +1.0), "q": (2, -1.0),
        }
        rotate = {
            "u": (0, +1.0), "o": (0, -1.0),
            "i": (1, +1.0), "k": (1, -1.0),
            "j": (2, +1.0), "l": (2, -1.0),
        }

        lower = key.lower()

        if lower in translate:
            axis, sign = translate[lower]
            delta = np.zeros(3, dtype=np.float64)
            delta[axis] = sign * self.step_pos
            if self.translate_world_frame:
                self.offset.translate_world(delta, self._last_q_mocap)
            else:
                self.offset.translate_local(delta)
            return

        if lower in rotate:
            axis, sign = rotate[lower]
            axis_vec = np.zeros(3, dtype=np.float64)
            axis_vec[axis] = 1.0
            self.offset.rotate_local(q_from_axis_angle(axis_vec, sign * self.step_rot))
            return

        if lower == "n":
            self.step_pos = max(self.step_pos * 0.5, 1.0e-5)
            self.step_rot = max(self.step_rot * 0.5, np.radians(0.005))
            return
        if lower == "m":
            self.step_pos = min(self.step_pos * 2.0, 0.5)
            self.step_rot = min(self.step_rot * 2.0, np.radians(45.0))
            return

        if lower == "f":
            self.translate_world_frame = not self.translate_world_frame
            self._note(
                "translation frame: "
                + ("mocap WORLD" if self.translate_world_frame else "mocap BODY")
            )
            return
        if lower == "g":
            self.solve_gravity_now()
            return
        if lower == "b":
            self.solve_ground_now()
            return
        if lower == "c":
            self.capture_sample()
            return
        if lower == "v":
            self.solve_from_samples()
            return
        if lower == "x":
            self.samples = []
            self._note("cleared captured samples")
            return
        if lower == "y":
            self.toggle_drive_recording()
            return
        if lower == "r":
            self.offset = self.initial_offset.copy()
            self._note("offset reset to the config value")
            return
        if lower == "z":
            self.offset = RootOffset(np.zeros(3), [1.0, 0.0, 0.0, 0.0])
            self._note("offset zeroed")
            return
        if lower == "p":
            self._print_offset()
            return
        if lower == "t":
            self.render_mode = (self.render_mode + 1) % 3
            self._note(
                "overlay: "
                + ["off", "frame triads", "frame triads + mocap markers"][self.render_mode]
            )
            return
        if lower == "?":
            sys.stdout.write("\r\033[K" + HELP_TEXT + "\n")
            sys.stdout.flush()
            return
        if key in ("\r", "\n"):
            self.save()
            return
        if key == "\x1b":  # Esc
            self._quit = True
            return

    # ------------------------------------------------------------------
    # Rendering + status
    # ------------------------------------------------------------------

    def draw_markers(self, viewer) -> None:
        scene = viewer.user_scn
        scene.ngeom = 0
        if self.render_mode == 0:
            return

        axis_len = float(self.cfg.get("triad_axis_len_m", 0.25))
        axis_radius = float(self.cfg.get("triad_axis_radius_m", 0.008))

        # Raw mocap rigid-body frame: thin, washed-out triad.
        self._draw_triad(
            scene, self._last_p_mocap, self._last_q_mocap,
            0.7 * axis_len, 0.5 * axis_radius, alpha=0.45,
        )
        _add_sphere(scene, self._last_p_mocap, 0.02, [1.0, 1.0, 1.0, 0.6])

        # Reconstructed root frame: full-strength triad.
        p_root, q_root = self.offset.apply(self._last_p_mocap, self._last_q_mocap)
        self._draw_triad(scene, p_root, q_root, axis_len, axis_radius, alpha=1.0)
        _add_sphere(scene, p_root, 0.025, [1.0, 1.0, 0.2, 0.95])

        # The offset itself, as a line from the mocap pivot to the root.
        _add_capsule(scene, self._last_p_mocap, p_root, 0.004, [1.0, 1.0, 0.2, 0.8])

        # IMU gravity, drawn from the root. It should hang straight down once
        # roll/pitch are right; any visible lean is the remaining error.
        if np.isfinite(self._imu_age):
            gravity_world = q_apply(q_root, self._gravity_base)
            _add_capsule(
                scene, p_root, p_root + 0.4 * gravity_world,
                0.006, [1.0, 0.4, 0.0, 0.9],
            )

        if self.render_mode < 2:
            return

        frame = self.mocap.latest_frame()
        if frame is None:
            return

        marker_set_name = self.cfg.get("mocap_marker_set_name") or self.rigid_body_name
        radius = float(self.cfg.get("marker_radius_m", 0.012))
        for name, points in frame.marker_sets.items():
            if name == "all":  # Motive's aggregate set; duplicates the rest.
                continue
            highlight = marker_set_name is not None and name == marker_set_name
            rgba = [0.1, 1.0, 1.0, 0.95] if highlight else [0.6, 0.8, 1.0, 0.7]
            for point in points:
                _add_sphere(scene, point, radius, rgba)

        for point in frame.unlabeled_markers:
            _add_sphere(scene, point, 0.8 * radius, [1.0, 0.6, 0.6, 0.6])

    def _draw_triad(self, scene, origin, quat, length, radius, alpha) -> None:
        colors = [
            [1.0, 0.2, 0.2, alpha],
            [0.2, 1.0, 0.2, alpha],
            [0.3, 0.4, 1.0, alpha],
        ]
        rot = q_to_mat(quat)
        for axis in range(3):
            _add_capsule(
                scene, origin, np.asarray(origin) + length * rot[:, axis],
                radius, colors[axis],
            )

    def status_line(self) -> str:
        pos = self.offset.pos
        roll, pitch, yaw = self.offset.rpy()

        if not self._mocap_ok:
            mocap = "MOCAP:LOST "
        elif not self._tracking_valid:
            mocap = "MOCAP:UNTRACKED"
        else:
            mocap = f"MOCAP:{1000.0 * self._mocap_age:4.0f}ms"

        imu = "IMU:--" if not np.isfinite(self._imu_age) else f"IMU:{1000.0 * self._imu_age:4.0f}ms"

        return (
            f"xyz[{pos[0]:+.4f} {pos[1]:+.4f} {pos[2]:+.4f}]m "
            f"rpy[{np.degrees(roll):+7.3f} {np.degrees(pitch):+7.3f} {np.degrees(yaw):+7.3f}]deg "
            f"| step {1000.0 * self.step_pos:.2f}mm/{np.degrees(self.step_rot):.3f}deg "
            f"{'WORLD' if self.translate_world_frame else 'BODY '} "
            f"| grav {self._gravity_residual_deg:6.3f}deg "
            f"gnd {1000.0 * (self._lowest_z - self.ground_z):+7.1f}mm "
            f"| {mocap} {imu} n={len(self.samples)}"
            f"{' REC' if self.drive_recording else ''}"
        )

    def _print_offset(self) -> None:
        roll, pitch, yaw = self.offset.rpy()
        inv_pos, inv_quat = self.offset.inverse()
        sys.stdout.write(
            "\r\033[K"
            "[CALIB] mocap rigid body -> base_link\n"
            f"          pos       [{self.offset.pos[0]:+.6f}, {self.offset.pos[1]:+.6f}, "
            f"{self.offset.pos[2]:+.6f}] m\n"
            f"          rpy       [{np.degrees(roll):+.4f}, {np.degrees(pitch):+.4f}, "
            f"{np.degrees(yaw):+.4f}] deg\n"
            f"          quat wxyz [{self.offset.quat[0]:+.8f}, {self.offset.quat[1]:+.8f}, "
            f"{self.offset.quat[2]:+.8f}, {self.offset.quat[3]:+.8f}]\n"
            "        base_link -> mocap rigid body\n"
            f"          pos       [{inv_pos[0]:+.6f}, {inv_pos[1]:+.6f}, {inv_pos[2]:+.6f}] m\n"
            f"          quat wxyz [{inv_quat[0]:+.8f}, {inv_quat[1]:+.8f}, "
            f"{inv_quat[2]:+.8f}, {inv_quat[3]:+.8f}]\n"
        )
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self) -> None:
        roll, pitch, yaw = self.offset.rpy()
        inv_pos, inv_quat = self.offset.inverse()
        grav_rms, ground_rms = (
            self._sample_residuals() if self.samples else (float("nan"), float("nan"))
        )

        payload = {
            "mocap_root_offset_pos": [float(v) for v in self.offset.pos],
            "mocap_root_offset_quat_wxyz": [float(v) for v in self.offset.quat],
            "mocap_root_offset_rpy": [float(roll), float(pitch), float(yaw)],
            "mocap_root_offset_rpy_deg": [
                float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))
            ],
            "root_to_mocap_pos": [float(v) for v in inv_pos],
            "root_to_mocap_quat_wxyz": [float(v) for v in inv_quat],
            "calibration_meta": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_config": os.path.abspath(self.cfg_path),
                "mujoco_xml_path": str(self.cfg["mujoco_xml_path"]),
                "mocap_rigid_body_id": int(self.rigid_body_id),
                "mocap_rigid_body_name": self.rigid_body_name,
                "mocap_up_axis": self.mocap.up_axis,
                "mocap_frame_rate_hz": self.mocap.frame_rate,
                "mocap_server": self.mocap.server_info.get("name"),
                "ground_z": float(self.ground_z),
                "num_samples": int(len(self.samples)),
                "gravity_residual_rms_deg": float(grav_rms),
                "ground_residual_rms_mm": float(ground_rms),
                "live_gravity_residual_deg": float(self._gravity_residual_deg),
                "live_ground_residual_mm": float(
                    1000.0 * (self._lowest_z - self.ground_z)
                ),
            },
        }

        header = (
            "# B2W mocap rigid body -> base_link (root) offset.\n"
            "#\n"
            "# Generated by deploy/b2w_mocap_root_calibration.py. Apply as:\n"
            "#\n"
            "#     p_root_world = p_mocap_world + R(q_mocap_world) @ offset_pos\n"
            "#     q_root_world = q_mocap_world (x) offset_quat_wxyz\n"
            "#\n"
            "# Positions are metres, quaternions are wxyz, and the mocap world\n"
            "# frame is Z-up (the client converts from Motive's up axis).\n"
        )

        path = os.path.abspath(self.output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            backup = path + ".bak"
            shutil.copyfile(path, backup)
            self._note(f"existing file backed up to {backup}")

        with open(path, "w") as f:
            f.write(header)
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)

        self._note(f"saved -> {path}")
        self._print_offset()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not self.keys.enabled:
            print(
                "[CALIB][WARN] stdin is not a terminal: the viewer will render "
                "but keyboard editing is disabled."
            )
        self.keys.start()

        print(HELP_TEXT)
        self._print_offset()

        render_period = 1.0 / max(float(self.cfg.get("render_hz", 30.0)), 1.0)
        status_period = 1.0 / max(float(self.cfg.get("status_hz", 10.0)), 1.0)
        next_status = 0.0

        with mujoco.viewer.launch_passive(self.scene.model, self.scene.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = [0.0, 0.0, 0.4]

            while viewer.is_running() and not self._quit:
                loop_start = time.perf_counter()

                # Telemetry first: a solver key pressed this iteration must act
                # on the pose that is on screen right now, not the previous
                # frame's -- on the very first iteration there is no previous
                # frame at all and the key would simply be refused.
                self.update_telemetry()

                for key in self.keys.drain():
                    self.handle_key(key)
                    if self._quit:
                        break

                if self.drive_recording and self._mocap_ok:
                    self.drive_samples.append(
                        (self._last_p_mocap.copy(), self._last_q_mocap.copy())
                    )

                self.refresh_scene()
                self.draw_markers(viewer)
                viewer.sync()

                now = time.perf_counter()
                if now >= next_status:
                    next_status = now + status_period
                    sys.stdout.write("\r\033[K" + self.status_line())
                    sys.stdout.flush()

                remaining = render_period - (time.perf_counter() - loop_start)
                if remaining > 0.0:
                    time.sleep(remaining)

        sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the B2W mocap-rigid-body -> root offset in MuJoCo."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="deploy/configs/b2w_mocap_root_calibration.yaml",
        help="calibration YAML",
    )
    parser.add_argument(
        "--no-dds",
        action="store_true",
        help="skip the LowState subscription (renders the fallback joint pose)",
    )
    parser.add_argument("--net", default=None, help="override robot_net_interface")
    parser.add_argument("--rigid-body-id", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.net is not None:
        cfg["robot_net_interface"] = args.net
    if args.rigid_body_id is not None:
        cfg["mocap_rigid_body_id"] = args.rigid_body_id
        cfg["mocap_rigid_body_name"] = None

    calibrator = MocapRootCalibrator(cfg, args.config, use_dds=not args.no_dds)
    try:
        calibrator.connect()
        calibrator.run()
    except KeyboardInterrupt:
        print("\n[CALIB] interrupted")
    finally:
        calibrator.close()


if __name__ == "__main__":
    main()
