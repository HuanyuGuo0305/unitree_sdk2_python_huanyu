#!/usr/bin/env python3
"""Offline tests for deploy/b2wz1_hl_transport_uan_mocap.py. No robot, no DDS.

Run from the repository root in BOTH environments (each skips what it cannot
run and says why):

    # sim2sim parity (needs mujoco + the read-only unitree_mujoco checkout)
    ~/miniconda3/envs/env_unitree_mujoco/bin/python -B \
        deploy/tests/test_b2wz1_hl_transport_uan_mocap.py

    # deploy env: Z1 SDK forward kinematics vs the training MJCF fixture
    # (sibling checkouts ../unitree_mujoco_huanyu | ../unitree_mujoco and
    #  ../z1_sdk_huanyu | ../z1_sdk are found automatically; override with
    #  TRANSPORT_MJ_ROOT / TRANSPORT_Z1_SDK)
    LD_LIBRARY_PATH=../z1_sdk_huanyu/lib ~/miniconda3/envs/env_unitree_sdk2/bin/python -B \
        deploy/tests/test_b2wz1_hl_transport_uan_mocap.py

Every DDS entry point is replaced by a guard that FAILS the test if called,
so a pass also proves that no robot command could have been published.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.dont_write_bytecode = True  # never write into the read-only reference repo

import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../.."))
DEPLOY = os.path.join(REPO, "deploy")
for _p in (REPO, DEPLOY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import b2wz1_hl_transport_uan_mocap as T  # noqa: E402
import unitree_sdk2py.core.channel as dds_channel  # noqa: E402
from utils.mocap_perception import MocapBodySelector  # noqa: E402
from utils.math import quat_from_rotmat_wxyz  # noqa: E402
from utils.mocap_frames import q_apply, q_from_rpy, q_to_mat  # noqa: E402
from utils.natnet_client import RigidBodySample  # noqa: E402
from utils.remote_controller import KeyMap, RemoteController  # noqa: E402

CFG_PATH = os.path.join(DEPLOY, "configs", "b2wz1_hl_transport_uan_mocap.yaml")
FIXTURE = os.path.join(HERE, "fixtures", "z1_disk_fk_reference.json")
def _first_dir(env_key, *candidates):
    """$env_key if set, else the first existing sibling checkout."""
    if os.environ.get(env_key):
        return os.path.abspath(os.environ[env_key])
    paths = [os.path.abspath(os.path.join(REPO, c)) for c in candidates]
    return next((c for c in paths if os.path.isdir(c)), paths[0])


MJ_ROOT = _first_dir("TRANSPORT_MJ_ROOT", "../unitree_mujoco_huanyu", "../unitree_mujoco")
MJ_SIM = os.path.join(MJ_ROOT, "simulate_python")
MJ_CFG = os.path.join(MJ_SIM, "configs", "b2wz1_hl_transport_uan.yaml")
Z1_SDK_LIB = os.path.join(
    _first_dir("TRANSPORT_Z1_SDK", "../z1_sdk_huanyu", "../z1_sdk"), "lib")
MJ_SCENE_XML = os.path.join(MJ_ROOT, "unitree_robots", "b2wz1", "b2wz1_disk_puck.xml")

HIGH_SHA = "db78304602566538ea099c3f3a0679bcb9b731067d4e036374ad03cc537b70c1"
WBC_SHA = "6b5aa5de8a438c07062ab5d9acd8e9d562ccb7a72f32aad72912f7707139a517"
# A DIFFERENT WBC checkpoint (2026-09-07 retrieval WBC) that must be refused.
WRONG_WBC_PATH = "exported_policies/b2wz1_wbc_uan_plb/policy_wbc_uan_3.onnx"
UAN_SHA = "db53ac18f9834d8a70f0fb37d068ce4f84607da2628d57181f0dcb299ca4c4eb"

try:
    import mujoco  # noqa: F401
    import mujoco.viewer  # noqa: F401

    HAVE_MUJOCO_VIEWER = True
except Exception:  # noqa: BLE001
    HAVE_MUJOCO_VIEWER = False

try:
    import mujoco  # noqa: F401,F811

    if MJ_SIM not in sys.path:
        sys.path.insert(0, MJ_SIM)
    from transport_experiment.controllers import HLMethod, decode_residual  # noqa: E402
    from transport_experiment.scenarios import deploy_rng_scenario  # noqa: E402
    from transport_experiment.sim import TransportSim  # noqa: E402
    from transport_experiment.sim import (  # noqa: E402
        build_keypoints_from_kp0_yaw_pitch_roll_plb as sim_keypoints,
    )
    from utilities.mujoco_helper import get_sensor_slice  # noqa: E402

    HAVE_SIM = True
    SIM_SKIP = ""
except Exception as exc:  # noqa: BLE001
    HAVE_SIM = False
    SIM_SKIP = f"MuJoCo sim2sim reference unavailable here ({type(exc).__name__}: {exc})"


# ----------------------------------------------------------------------
# Guards: any DDS use fails the run
# ----------------------------------------------------------------------

_DDS_PATCHES = []


def _forbidden(*_a, **_k):
    raise AssertionError("A test tried to open DDS / publish to the robot.")


def setUpModule():  # noqa: N802
    for name in ("ChannelFactoryInitialize", "ChannelPublisher", "ChannelSubscriber"):
        p = mock.patch.object(dds_channel, name, side_effect=_forbidden)
        p.start()
        _DDS_PATCHES.append(p)


def tearDownModule():  # noqa: N802
    for p in _DDS_PATCHES:
        p.stop()


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def filled_cfg(**overrides):
    """Shipped YAML with the [REQUIRED SETUP] keys filled by SYNTHETIC values."""
    cfg = load_cfg()
    cfg.update(
        mocap_puck_name="TEST_PUCK_SYNTHETIC",
        mocap_puck_id=99,
        mocap_disk_name="TEST_DISK_SYNTHETIC",
        mocap_disk_id=98,
        puck_geometry_verified=True,
        disk_mount_verified=True,
        visualizer_enabled=False,
        hl_log_enabled=False,
        debug_print_period_steps=0,
    )
    cfg.update(overrides)
    return cfg


def placeholder_cfg():
    """Shipped YAML with the operator-supplied Motive assets blanked out."""
    cfg = load_cfg()
    cfg.update(mocap_puck_name=None, mocap_puck_id=None, mocap_disk_name=None, mocap_disk_id=None)
    return cfg


def write_cfg(cfg, tmpdir):
    path = os.path.join(tmpdir, "transport_test.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def load_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Hardware fakes
# ----------------------------------------------------------------------


class FakeZ1:
    """Static Z1: default q, gripperStator pose from the MJCF fixture."""

    def __init__(self, cfg):
        s = load_fixture()["samples"][0]  # default arm configuration
        self.arm_kps_runtime = np.asarray(cfg["arm_kps_runtime"], dtype=np.float32)
        self.arm_kds_runtime = np.asarray(cfg["arm_kds_runtime"], dtype=np.float32)
        self.unitree_arm_interface = types.SimpleNamespace(
            ArmFSMState=types.SimpleNamespace(PASSIVE="PASSIVE")
        )
        self.fsm = "LOWCMD"
        self.q = np.asarray(s["q"], dtype=np.float32)
        self.qd = np.zeros(6, dtype=np.float32)
        self.gripper_q = 0.0
        self.stator_pos = np.asarray(s["stator_pos_b"], dtype=np.float32)
        self.stator_rot = np.asarray(s["stator_rot_b"], dtype=np.float32)
        self.sent = []

    def read_state(self):
        pass

    def get_fsm_state(self):
        return self.fsm

    def compute_policy_ee_pose_in_base(self):
        return self.stator_pos.copy(), self.stator_rot.copy()

    def track_target_pd_runtime_once(self, **kw):
        self.sent.append(kw)


class FakePerception:
    """Mocap snapshot with the puck placed in the DISK frame of FakeZ1.

    The robot stands at an arbitrary world pose (yawed, offset), so the mocap
    puck/disk path really goes through world coordinates. The mocap disk is
    the FK disk unless an error (disk_err_pos_d [m, disk axes],
    disk_err_rpy [rad, about disk axes]) is injected.
    """

    def __init__(self, z1, disk, puck_d=(0.0, 0.0, 0.0425), base_height=0.6):
        self.z1, self.disk = z1, disk
        self.puck_d = np.asarray(puck_d, dtype=np.float64)
        self.base_height = base_height
        self.puck_valid, self.puck_reason = True, "OK"
        self.disk_valid, self.disk_reason = True, "OK"
        self.body_valid = True
        self.disk_err_pos_d = np.zeros(3)
        self.disk_err_rpy = np.zeros(3)
        self.p_root = np.array([1.0, 2.0, base_height])
        self.q_root = q_from_rpy(0.0, 0.0, 0.7)

    def fk_disk_base(self):
        c, R = self.disk.frame_in(self.z1.stator_pos, self.z1.stator_rot)
        return c.astype(np.float64), R.astype(np.float64)

    def puck_base(self):
        c, R = self.fk_disk_base()
        return c + R @ self.puck_d

    def get_latest_snapshot(self):
        R_root = q_to_mat(self.q_root)
        c, R = self.fk_disk_base()
        disk_w = self.p_root + R_root @ (c + R @ self.disk_err_pos_d)
        R_dw = R_root @ R @ q_to_mat(q_from_rpy(*self.disk_err_rpy))
        puck_w = self.p_root + R_root @ self.puck_base()
        snap = {
            "puck": {"valid": self.puck_valid, "reason": self.puck_reason, "age_ms": 2.0,
                     "position_world": puck_w.tolist() if self.puck_valid else None,
                     "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                     "position_base": self.puck_base().tolist()
                     if self.puck_valid and self.body_valid else None},
            "disk": {"valid": self.disk_valid, "reason": self.disk_reason, "age_ms": 2.0,
                     "position_world": disk_w.tolist() if self.disk_valid else None,
                     "quat_wxyz": quat_from_rotmat_wxyz(R_dw).tolist() if self.disk_valid else None},
        }
        if not self.body_valid:
            snap.update(body={"valid": False},
                        base_height={"valid": False, "reason": "BODY_STALE"})
            return snap
        snap.update(
            body={"valid": True, "age_ms": 2.0, "root_position_world": self.p_root.tolist(),
                  "root_quat_wxyz": self.q_root.tolist()},
            base_height={"valid": True, "height_m": self.base_height, "reason": "OK"},
        )
        return snap


def static_low_state(cfg, gyro=(0.0, 0.0, 0.0)):
    """Standing at the default pose, level."""
    default = np.asarray(cfg["default_b2w_pos_policy"], dtype=np.float32)
    policy_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
    ]
    hw_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
    ]
    motors = [types.SimpleNamespace(q=float(default[policy_names.index(n)]), dq=0.0) for n in hw_names]
    return types.SimpleNamespace(
        tick=1,
        imu_state=types.SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=list(gyro)),
        motor_state=motors,
    )


def make_static_controller(cfg_overrides=None, **perc_kwargs):
    cfg = filled_cfg(**(cfg_overrides or {}))
    tmp = tempfile.mkdtemp(prefix="transport_test_")
    path = write_cfg(cfg, tmp)
    z1 = FakeZ1(cfg)
    disk = T.DiskGeometry(cfg["disk_center_offset_gripper"], cfg["disk_x_axis_gripper"],
                          cfg["disk_y_axis_gripper"], cfg["disk_z_axis_gripper"],
                          cfg["disk_thickness"], cfg["disk_radius"])
    perception = FakePerception(z1, disk, **perc_kwargs)
    with redirect_stdout(io.StringIO()):
        ctrl = T.B2WZ1TransportMocapController(path, z1=z1, perception=perception, connect_dds=False)
    ctrl.low_state = static_low_state(cfg)
    ctrl.remote_controller = RemoteController()
    return ctrl, z1, perception


# ======================================================================
# 1. Models
# ======================================================================


class TestModels(unittest.TestCase):
    def setUp(self):
        self.cfg = load_cfg()

    def test_pinned_hashes_and_shapes(self):
        import onnxruntime as ort

        for key, sha_key, shape_in, shape_out in (
            ("high_policy_path", "high_policy_sha256", 192, 6),
            ("low_policy_path", "low_policy_sha256", 400, 22),
        ):
            path = os.path.join(REPO, self.cfg[key])
            self.assertEqual(T.file_sha256(path), self.cfg[sha_key], key)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.assertEqual(sess.get_inputs()[0].shape[-1], shape_in)
            self.assertEqual(sess.get_outputs()[0].shape[-1], shape_out)
        self.assertEqual(self.cfg["high_policy_sha256"], HIGH_SHA)
        self.assertEqual(self.cfg["low_policy_sha256"], WBC_SHA)

    @unittest.skipUnless(os.path.isfile(os.path.join(REPO, WRONG_WBC_PATH)), "other WBC not present")
    def test_wrong_transport_wbc_is_rejected(self):
        """Any WBC other than the one the transport policy was trained on is refused."""
        self.assertNotEqual(T.file_sha256(os.path.join(REPO, WRONG_WBC_PATH)), WBC_SHA)
        cfg = filled_cfg(low_policy_path=WRONG_WBC_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                T.B2WZ1TransportMocapController(write_cfg(cfg, tmp), z1=FakeZ1(cfg),
                                                perception=object(), connect_dds=False)

    def test_high_policy_is_deterministic_and_bounded(self):
        import onnxruntime as ort

        sess = ort.InferenceSession(os.path.join(REPO, self.cfg["high_policy_path"]),
                                    providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        self.assertEqual(sess.get_inputs()[0].shape[0], 1)  # exported with batch 1
        for _ in range(64):
            obs = (3.0 * rng.normal(size=(1, 192))).astype(np.float32)
            a1 = sess.run(None, {"obs": obs})[0]
            a2 = sess.run(None, {"obs": obs})[0]
            np.testing.assert_array_equal(a1, a2)
            self.assertTrue(np.all(np.abs(a1) <= 1.0))

    @unittest.skipUnless(os.path.isdir(MJ_ROOT), "unitree_mujoco checkout not present")
    def test_same_checkpoints_as_sim2sim(self):
        with open(MJ_CFG) as f:
            mj = yaml.safe_load(f)
        self.assertEqual(T.file_sha256(os.path.join(MJ_ROOT, mj["high_policy_path"])), HIGH_SHA)
        self.assertEqual(T.file_sha256(os.path.join(MJ_ROOT, mj["low_policy_path"])), WBC_SHA)
        self.assertEqual(T.file_sha256(os.path.join(MJ_ROOT, mj["uan_model_path"])), UAN_SHA)


# ======================================================================
# 2. Config / placeholders / sim2sim constant parity
# ======================================================================


class TestConfig(unittest.TestCase):
    def test_shipped_yaml_refuses_until_setup_done(self):
        problems = T.validate_transport_setup(placeholder_cfg())
        joined = "\n".join(problems)
        for key in ("mocap_puck_name", "mocap_disk_name"):
            self.assertIn(key, joined)
        for gone in ("puck_mass_kg", "mocap_puck_offset_local"):
            self.assertNotIn(gone, joined)
        self.assertEqual(T.validate_transport_setup(filled_cfg()), [])
        # FK disk source: no disk rigid body needed.
        self.assertEqual(T.validate_transport_setup(filled_cfg(
            disk_frame_source="fk", mocap_disk_name=None, mocap_disk_id=None)), [])
        # Offsets are optional (pivots placed in Motive), as in retrieval.
        cfg = filled_cfg()
        cfg.pop("mocap_puck_offset_local", None)
        cfg.pop("mocap_disk_offset_local", None)
        self.assertEqual(T.validate_transport_setup(cfg), [])

    def test_invalid_setup_values(self):
        cases = {
            "puck must be a different": dict(mocap_puck_id=9),
            "disk must be a different rigid body from the puck": dict(
                mocap_disk_name="TEST_PUCK_SYNTHETIC"),
            "disk must be a different rigid body from the body": dict(mocap_disk_id=9),
            "mocap_disk_max_age_s": dict(mocap_disk_max_age_s=0.3),
            "disk_frame_source": dict(disk_frame_source="tag"),
            "puck_geometry_verified": dict(puck_geometry_verified=False),
            "disk_mount_verified": dict(disk_mount_verified=False),
            "mocap_puck_offset_local": dict(mocap_puck_offset_local=[0.0, float("nan"), 0.0]),
            "simulation-only": dict(uan_model_path="x.pt"),
            "mocap_puck_max_age_s": dict(mocap_puck_max_age_s=0.3),
            "command_scale": dict(command_scale=[0.8, 0.6, 0.6]),
            "mocap root frame": dict(mocap_root_frame_is_base_link=False),
        }
        for needle, override in cases.items():
            with self.subTest(needle):
                self.assertIn(needle, "\n".join(T.validate_transport_setup(filled_cfg(**override))))

    def test_controller_refuses_placeholders_before_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "refusing to start"):
                T.B2WZ1TransportMocapController(write_cfg(placeholder_cfg(), tmp))

    def test_main_exits_before_dds(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["prog", "lo", write_cfg(placeholder_cfg(), tmp)]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    T.main()
        self.assertEqual(ctx.exception.code, 2)

    def test_cli_disk_frame_source_override(self):
        cfg = placeholder_cfg()
        cfg.update(mocap_puck_name="TEST_PUCK", mocap_puck_id=99)  # disk asset still missing
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cfg(cfg, tmp)
            for source, expect_exit in (("mocap", True), ("fk", False)):
                argv = ["prog", "lo", path, "--disk-frame-source", source]
                with self.subTest(source), mock.patch.object(sys, "argv", argv), \
                        redirect_stdout(io.StringIO()):
                    if expect_exit:
                        with self.assertRaises(SystemExit):
                            T.main()
                    else:
                        # Validation passes; the first thing after it is DDS,
                        # which the test guard turns into an AssertionError.
                        with self.assertRaisesRegex(AssertionError, "DDS"):
                            T.main()

    def test_shipped_yaml_is_complete(self):
        self.assertEqual(T.validate_transport_setup(load_cfg()), [])

    @unittest.skipUnless(os.path.isfile(MJ_CFG), "sim2sim YAML not present")
    def test_task_constants_equal_sim2sim(self):
        with open(MJ_CFG) as f:
            mj = yaml.safe_load(f)
        cfg = load_cfg()
        keys = [
            "ll_history_length", "ll_obs_dim_per_step", "ll_obs_dim", "ll_action_dim",
            "hl_history_length", "hl_obs_dim_per_step", "hl_obs_dim", "hl_action_dim",
            "leg_action_scale", "arm_action_scale", "wheel_action_scale", "gripper_hold_pos",
            "ground_z", "ee_kp_dx", "ee_kp_dz", "base_cmd_scale", "kp0_x_range", "kp0_y_range",
            "kp0_z_range", "ee_yaw_range", "ee_pitch_range", "ee_roll_range",
            "ee_pos_action_residual_scale", "ee_rot_action_residual_scale",
            "reset_settle_time_s", "ee_ref_resample_time_range_s", "ee_fast_transition_time_s",
            "ee_low_transition_time_s", "canonical_ee_positions", "canonical_ee_yaw",
            "canonical_ee_pitch", "canonical_ee_roll", "base_ref_ranges",
            "disk_center_offset_gripper", "disk_x_axis_gripper", "disk_y_axis_gripper",
            "disk_z_axis_gripper", "disk_thickness", "puck_radius", "puck_height",
            "puck_spawn_xy_randomization", "puck_spawn_clearance", "puck_drop_no_contact_steps",
        ]
        for k in keys:
            with self.subTest(k):
                self.assertEqual(cfg[k], mj[k])
        # default joint pose: sim2sim MuJoCo order -> policy order
        dj = np.asarray(mj["default_joint_pos"])
        mj_names = ["FL_hip", "FL_thigh", "FL_calf", "FL_wheel", "FR_hip", "FR_thigh", "FR_calf",
                    "FR_wheel", "RL_hip", "RL_thigh", "RL_calf", "RL_wheel", "RR_hip", "RR_thigh",
                    "RR_calf", "RR_wheel", "j1", "j2", "j3", "j4", "j5", "j6", "jg"]
        legs = [f"{l}_{p}" for p in ("hip", "thigh", "calf") for l in ("FL", "FR", "RL", "RR")]
        policy = [dj[mj_names.index(n)] for n in legs] + list(dj[16:22])
        np.testing.assert_allclose(cfg["default_joint_pos_policy"], policy)
        # timing: 50 Hz WBC, 10 Hz HL
        self.assertAlmostEqual(mj["simulation_dt"] * mj["ll_control_decimation"], cfg["control_dt"])
        self.assertEqual(mj["ll_steps_per_hl_step"], cfg["ll_steps_per_hl_step"])


# ======================================================================
# 3. Reference process / decode (pure)
# ======================================================================


class TestTransportTask(unittest.TestCase):
    def setUp(self):
        self.cfg = load_cfg()
        self.task = T.TransportTask(self.cfg, 0.1)
        self.high = np.array([0.55, 0.0, 1.10], dtype=np.float32)

    def _settle(self, base=(0.0, 0.0, 0.0)):
        self.task.reset(self.high)
        for _ in range(self.task.settle_steps):
            self.task.advance(lambda: self.high, np.asarray(base, dtype=np.float32))

    def test_settle_masks_residual_and_holds_kp0(self):
        self.task.reset(self.high)
        self.assertTrue(self.task.settling)
        self.task.decode(np.ones(6, dtype=np.float32))
        np.testing.assert_array_equal(self.task.kp0_cmd, self.high)
        np.testing.assert_array_equal(self.task.executed_action, np.zeros(6, dtype=np.float32))
        np.testing.assert_array_equal(self.task.raw_action, np.ones(6, dtype=np.float32))
        np.testing.assert_array_equal(self.task.base_command, np.zeros(3, dtype=np.float32))

    def test_settle_end_remeasures_high_and_takes_base(self):
        self.task.reset(self.high)
        settled = np.array([0.56, 0.01, 1.09], dtype=np.float32)
        for _ in range(self.task.settle_steps - 1):
            self.task.advance(lambda: settled, np.array([0.3, 0, 0], dtype=np.float32))
            self.assertTrue(self.task.settling)
            np.testing.assert_array_equal(self.task.base_velocity_ref, np.zeros(3))
        self.task.advance(lambda: settled, np.array([0.3, 0, 0], dtype=np.float32))
        self.assertFalse(self.task.settling)
        np.testing.assert_array_equal(self.task.episode_high_kp0, settled)
        np.testing.assert_array_equal(self.task.ee_kp0_ref, settled)
        np.testing.assert_allclose(self.task.base_velocity_ref, [0.3, 0, 0])

    def test_low_transition_min_jerk_and_min_hold(self):
        self._settle()
        self.task.request_target(1)
        self.task.advance(lambda: self.high, np.zeros(3, dtype=np.float32))
        self.assertIn("REJECTED", self.task.last_event)  # 1 step after settle end
        for _ in range(self.task.min_hold_steps - 2):
            self.task.advance(lambda: self.high, np.zeros(3, dtype=np.float32))
        self.task.request_target(1)
        self.task.advance(lambda: self.high, np.zeros(3, dtype=np.float32))
        self.assertEqual(self.task.last_event, "target -> LOW")
        np.testing.assert_array_equal(self.task.ee_kp0_ref, self.high)  # tau = 0
        target = np.asarray(self.cfg["canonical_ee_positions"][1], dtype=np.float32)
        path = []
        for k in range(1, self.task.low_steps + 3):
            self.task.advance(lambda: self.high, np.zeros(3, dtype=np.float32))
            path.append(self.task.ee_kp0_ref.copy())
            expect = self.high + T.minimum_jerk_blend(min(k, 20) / 20.0) * (target - self.high)
            np.testing.assert_allclose(path[-1], expect, atol=1e-7)
        np.testing.assert_allclose(path[self.task.low_steps - 1], target, atol=1e-7)

    def test_base_reference_clipped_to_training_range(self):
        self._settle()
        self.task.advance(lambda: self.high, np.array([2.0, -2.0, 0.1], dtype=np.float32))
        np.testing.assert_allclose(self.task.base_velocity_ref, [0.6, -0.6, 0.1], atol=1e-7)
        with self.assertRaises(FloatingPointError):
            self.task.advance(lambda: self.high, np.array([np.nan, 0, 0], dtype=np.float32))

    def test_decode_rejects_nonfinite(self):
        self._settle()
        with self.assertRaises(FloatingPointError):
            self.task.decode(np.array([0, 0, np.inf, 0, 0, 0], dtype=np.float32))


class TestPuckGeometry(unittest.TestCase):
    def setUp(self):
        cfg = load_cfg()
        self.disk = T.DiskGeometry(cfg["disk_center_offset_gripper"], cfg["disk_x_axis_gripper"],
                                   cfg["disk_y_axis_gripper"], cfg["disk_z_axis_gripper"],
                                   cfg["disk_thickness"], cfg["disk_radius"])
        self.mon = T.PuckContactMonitor(0.1, 0.042, 0.01, 5)

    def test_puck_d_round_trip_any_frame(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            q = rng.normal(size=4)
            R = q_to_mat(q / np.linalg.norm(q)).astype(np.float32)
            p = rng.normal(size=3).astype(np.float32)
            d = rng.uniform(-0.1, 0.1, size=3)
            c, Rd = self.disk.frame_in(p, R)
            puck = c + Rd @ d
            np.testing.assert_allclose(self.disk.puck_pos_disk(puck, p, R), d, atol=2e-6)

    def test_contact_proxy_and_debounce(self):
        on = np.array([0.02, -0.03, 0.042])
        self.assertTrue(self.mon.in_contact(on))
        self.assertFalse(self.mon.in_contact(on + [0, 0, 0.011]))     # lifted
        self.assertFalse(self.mon.in_contact([0.08, 0.07, 0.042]))     # centre off the rim
        self.assertFalse(self.mon.in_contact([np.nan, 0, 0.042]))
        self.assertFalse(self.mon.in_contact(None))
        for k in range(1, 5):
            self.assertEqual(self.mon.update([0.2, 0, 0.042], settling=False), (False, k, False))
        self.assertEqual(self.mon.update([0.2, 0, 0.042], settling=False), (False, 5, True))
        self.mon.reset()
        self.mon.update([0.2, 0, 0.042], settling=False)
        self.assertEqual(self.mon.update(on, settling=False), (True, 0, False))    # landed: reset
        self.assertEqual(self.mon.update([0.2, 0, 0.0], settling=True), (False, 0, False))

    def test_start_gate(self):
        ok = lambda d: T.puck_start_gate(d, 0.01, 0.0005, 0.01, 0.042)[0]  # noqa: E731
        self.assertTrue(ok([0.0, 0.0, 0.0425]))
        self.assertTrue(ok([0.019, -0.019, 0.045]))
        self.assertFalse(ok([0.021, 0.0, 0.0425]))
        self.assertFalse(ok([0.0, 0.0, 0.054]))
        self.assertFalse(ok(None))
        self.assertFalse(ok([np.nan, 0.0, 0.04]))


# ======================================================================
# 4. OptiTrack perception (fake NatNet stream)
# ======================================================================


class FakeNatNet:
    def __init__(self):
        self.samples, self.names = {}, {}
        self.frame = types.SimpleNamespace(frame_number=7)
        self.frame_count, self.parse_error_count = 100, 0

    def start(self):
        pass

    def stop(self):
        pass

    def wait_for_first_frame(self, timeout_s):
        return True

    def rigid_body_ids(self):
        return sorted(self.samples)

    def rigid_body_names(self):
        return dict(self.names)

    def latest_rigid_body(self, rb_id):
        return self.samples.get(rb_id)

    def latest_frame(self):
        return self.frame


def _rb(rb_id, pos, quat=(1, 0, 0, 0), valid=True, age=0.0):
    return RigidBodySample(rb_id, np.asarray(pos, float), np.asarray(quat, float), 0.0005,
                           valid, 7, time.monotonic() - age)


class TestMocap(unittest.TestCase):
    DISK_POS = np.array([1.45, 2.25, 1.15])
    DISK_Q = q_from_rpy(0.3, -0.2, 1.1)

    def make(self, puck_offset=(0.0, 0.0, 0.0), puck_id=20, with_disk=True):
        perc = T.TransportMocapPerception(
            body=MocapBodySelector("body", "B2", 9, [0, 0, 0], 0.25),
            puck_body=MocapBodySelector("puck", "PUCK", puck_id, list(puck_offset), 0.10),
            disk_body=MocapBodySelector("disk", "DISK", 21, [0, 0, 0], 0.10) if with_disk else None,
            root_frame_is_base_link=True, verbose=False,
        )
        fake = FakeNatNet()
        fake.names = {9: "B2", puck_id: "PUCK", 21: "DISK"}
        yaw = 0.7
        self.q_body = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        fake.samples[9] = _rb(9, [1.0, 2.0, 0.6], self.q_body)
        fake.samples[puck_id] = _rb(puck_id, [1.5, 2.3, 1.2], [0.0, 1.0, 0.0, 0.0])
        fake.samples[21] = _rb(21, self.DISK_POS, self.DISK_Q)
        perc.client = fake
        perc.initialize()
        return perc, fake

    def test_disk_channel_and_puck_d_from_world_poses(self):
        perc, fake = self.make()
        s = perc.get_latest_snapshot()
        self.assertTrue(s["valid"])
        np.testing.assert_allclose(s["disk"]["position_world"], self.DISK_POS, atol=1e-12)
        R = q_to_mat(self.DISK_Q)
        d = T.puck_pos_in_frame(s["puck"]["position_world"], s["disk"]["position_world"],
                                q_to_mat(np.asarray(s["disk"]["quat_wxyz"])))
        np.testing.assert_allclose(d, R.T @ (np.array([1.5, 2.3, 1.2]) - self.DISK_POS), atol=1e-6)
        # Disk loss invalidates the snapshot with its own reason.
        fake.samples[21] = _rb(21, self.DISK_POS, self.DISK_Q, valid=False)
        s = perc.get_latest_snapshot()
        self.assertFalse(s["valid"])
        self.assertEqual(s["disk"]["reason"], "UNTRACKED")
        fake.samples[21] = _rb(21, self.DISK_POS, self.DISK_Q, age=0.15)
        self.assertEqual(perc.get_latest_snapshot()["disk"]["reason"], "STALE")

    def test_fk_mode_has_no_disk_channel(self):
        perc, _ = self.make(with_disk=False)
        s = perc.get_latest_snapshot()
        self.assertTrue(s["valid"])
        self.assertNotIn("disk", s)

    def test_valid_transform_and_offset(self):
        perc, fake = self.make(puck_offset=(0.0, 0.0, -0.05))
        s = perc.get_latest_snapshot()
        self.assertTrue(s["valid"])
        puck_w = np.array([1.5, 2.3, 1.2]) + q_apply(np.array([0.0, 1.0, 0.0, 0.0]), [0, 0, -0.05])
        expect = q_to_mat(self.q_body).T @ (puck_w - [1.0, 2.0, 0.6])
        np.testing.assert_allclose(s["puck"]["position_base"], expect, atol=1e-12)
        self.assertAlmostEqual(s["base_height"]["height_m"], 0.6)
        self.assertAlmostEqual(puck_w[2], 1.25)  # offset applied in the PUCK frame (flipped)

    def test_missing_untracked_stale(self):
        perc, fake = self.make()
        fake.samples[20] = _rb(20, [1.5, 2.3, 1.2], valid=False)
        self.assertEqual(perc.get_latest_snapshot()["puck"]["reason"], "UNTRACKED")
        fake.samples[20] = _rb(20, [1.5, 2.3, 1.2], age=0.15)
        s = perc.get_latest_snapshot()
        self.assertFalse(s["valid"])
        self.assertEqual(s["puck"]["reason"], "STALE")
        del fake.samples[20]
        self.assertEqual(perc.get_latest_snapshot()["puck"]["reason"], "NOT_IN_STREAM")
        fake.samples[20] = _rb(20, [1.5, 2.3, 1.2])
        fake.samples[9] = _rb(9, [1.0, 2.0, 0.6], age=1.0)
        s = perc.get_latest_snapshot()
        self.assertFalse(s["valid"])
        self.assertFalse(s["base_height"]["valid"])
        self.assertEqual(s["base_height"]["reason"], "BODY_STALE")
        # World poses survive a lost B2 (visualisation); the base frame does not.
        self.assertTrue(s["puck"]["valid"])
        self.assertIsNone(s["puck"]["position_base"])

    def test_initialize_requires_both_bodies_and_distinct(self):
        perc = T.TransportMocapPerception(
            body=MocapBodySelector("body", "B2", 9), puck_body=MocapBodySelector("puck", None, 20),
            root_frame_is_base_link=True, verbose=False)
        fake = FakeNatNet()
        fake.names = {9: "B2"}
        fake.samples[9] = _rb(9, [0, 0, 0.6])
        perc.client = fake
        with self.assertRaisesRegex(RuntimeError, "puck"):
            perc.initialize()
        perc = T.TransportMocapPerception(
            body=MocapBodySelector("body", "B2", 9), puck_body=MocapBodySelector("puck", "B2", None),
            root_frame_is_base_link=True, verbose=False)
        perc.client = fake
        with self.assertRaisesRegex(RuntimeError, "SAME rigid body"):
            perc.initialize()
        fake.names[20] = "PUCK"
        fake.samples[20] = _rb(20, [0, 0, 1.0])
        perc = T.TransportMocapPerception(
            body=MocapBodySelector("body", "B2", 9), puck_body=MocapBodySelector("puck", "PUCK", None),
            disk_body=MocapBodySelector("disk", None, 22), root_frame_is_base_link=True, verbose=False)
        perc.client = fake
        with self.assertRaisesRegex(RuntimeError, "disk"):
            perc.initialize()
        perc = T.TransportMocapPerception(
            body=MocapBodySelector("body", "B2", 9), puck_body=MocapBodySelector("puck", "PUCK", None),
            disk_body=MocapBodySelector("disk", "PUCK", None), root_frame_is_base_link=True,
            verbose=False)
        perc.client = fake
        with self.assertRaisesRegex(RuntimeError, "SAME rigid body"):
            perc.initialize()

    def test_natnet_parses_only_the_newest_queued_frame(self):
        """A backlog in the socket is skipped, never parsed oldest-first."""
        import socket as _socket
        import struct as _struct
        import threading as _threading

        from utils import natnet_client

        client = natnet_client.NatNetClient(local_ip="127.0.0.1", verbose=False)
        rx = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        tx = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        for i in range(50):  # backlog queued before the reader runs
            body = _struct.pack("<i", i)
            tx.sendto(_struct.pack("<HH", natnet_client.NAT_FRAMEOFDATA, len(body)) + body,
                      rx.getsockname())
        tx.sendto(_struct.pack("<HH", natnet_client.NAT_KEEPALIVE, 0), rx.getsockname())
        time.sleep(0.05)
        parsed = []

        def fake_decode(payload, recv_time=None):
            parsed.append((_struct.unpack("<i", payload)[0], recv_time))
            return types.SimpleNamespace(frame_number=parsed[-1][0], rigid_bodies={})

        client._decode_frame = fake_decode
        client._cmd_sock, client._data_sock = rx, None
        t0 = time.monotonic()
        thread = _threading.Thread(target=client._run, daemon=True)
        thread.start()
        time.sleep(0.3)
        client._stop_event.set()
        thread.join(timeout=2.0)
        rx.close()
        tx.close()
        self.assertEqual([p[0] for p in parsed], [49])
        self.assertEqual(client.skipped_frame_count, 49)
        self.assertGreaterEqual(parsed[0][1], t0)  # stamped at receive, not at send/queue time
        self.assertEqual(client.latest_frame().frame_number, 49)

    def test_optional_disk_is_dropped_when_not_streamed(self):
        perc = T.TransportMocapPerception(
            body=MocapBodySelector("body", "B2", 9), puck_body=MocapBodySelector("puck", "PUCK", 20),
            disk_body=MocapBodySelector("disk", "DISK", 21), disk_required=False,
            root_frame_is_base_link=True, verbose=False)
        fake = FakeNatNet()
        fake.names = {9: "B2", 20: "PUCK"}
        fake.samples[9] = _rb(9, [0, 0, 0.6])
        fake.samples[20] = _rb(20, [0.5, 0, 1.0])
        perc.client = fake
        perc.initialize()
        self.assertIsNone(perc.disk_body)
        s = perc.get_latest_snapshot()
        self.assertTrue(s["valid"])
        self.assertNotIn("disk", s)
        self.assertNotIn("markers", s)  # marker cloud is pulled by the view only
        # Optional disk present but untracked: snapshot stays valid.
        perc2, fake2 = self.make()
        perc2.disk_required = False
        fake2.samples[21] = _rb(21, self.DISK_POS, self.DISK_Q, valid=False)
        s = perc2.get_latest_snapshot()
        self.assertTrue(s["valid"])
        self.assertFalse(s["disk"]["valid"])

    def test_natnet_rejects_non_unit_quaternion(self):
        """Invalid quaternions never reach a snapshot: the parser drops them."""
        import inspect

        from utils import natnet_client

        src = inspect.getsource(natnet_client)
        self.assertIn("non-unit rigid-body quaternion", src)


# ======================================================================
# 5. Controller lifecycle / safety with static fakes (real ONNX policies)
# ======================================================================


class TestControllerSafety(unittest.TestCase):
    def setUp(self):
        self.ctrl, self.z1, self.perc = make_static_controller()
        with redirect_stdout(io.StringIO()):
            self.ctrl.initialize_policy_state_and_history()

    def steps(self, n):
        out = None
        with redirect_stdout(io.StringIO()):
            for _ in range(n):
                out = self.ctrl.wbc_step()
                if out is not None:
                    return out
        return out

    def test_nominal_static_run_has_no_fault(self):
        self.assertIsNone(self.steps(5 * 30))
        self.assertFalse(self.ctrl.task.settling)
        self.assertTrue(np.all(np.abs(self.ctrl.task.executed_action) <= 1.0))
        self.assertEqual(self.z1.sent, [])  # wbc_step never actuates by itself

    def test_select_and_passive(self):
        self.ctrl.remote_controller.button[KeyMap.select] = 1
        self.assertEqual(self.steps(1), "SELECT pressed")
        self.ctrl.remote_controller.button[KeyMap.select] = 0
        self.z1.fsm = "PASSIVE"
        self.assertEqual(self.steps(1), "Z1 entered PASSIVE state")

    def test_mocap_loss_is_immediate(self):
        self.assertIsNone(self.steps(12))
        for valid, reason in ((False, "UNTRACKED"), (False, "STALE"), (False, "NOT_IN_STREAM")):
            self.perc.puck_valid, self.perc.puck_reason = valid, reason
            self.assertIn(f"PUCK:{reason}", self.steps(1))
        self.perc.puck_valid = True
        self.perc.body_valid = False
        self.assertIn("BASE_HEIGHT", self.steps(1))

    def test_puck_drop_after_training_debounce(self):
        self.assertIsNone(self.steps(5 * 6))  # past the settle
        self.perc.puck_d = np.array([0.15, 0.0, 0.0425])  # centre beyond the rim
        n = 0
        with redirect_stdout(io.StringIO()):
            while True:
                n += 1
                r = self.ctrl.wbc_step()
                if r is not None:
                    break
        self.assertIn("PUCK DROP", r)
        self.assertLessEqual(n, 5 * 5 + 1)
        self.assertGreater(n, 5 * 4)

    def test_transient_jump_does_not_drop(self):
        self.assertIsNone(self.steps(5 * 6))
        self.perc.puck_d = np.array([0.0, 0.0, 0.30])
        self.assertIsNone(self.steps(5 * 2))
        self.perc.puck_d = np.array([0.0, 0.0, 0.0425])
        self.assertIsNone(self.steps(5 * 10))
        self.assertEqual(self.ctrl.contact_monitor.no_contact_steps, 0)

    def test_nonfinite_observation_raises(self):
        self.assertIsNone(self.steps(3))
        self.ctrl.low_state = static_low_state(self.ctrl.cfg, gyro=(np.nan, 0.0, 0.0))
        with self.assertRaises(FloatingPointError):
            self.steps(1)

    def test_hl_csv_log_and_step_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.ctrl.hl_log_enabled, self.ctrl.hl_log_dir = True, tmp
            with redirect_stdout(io.StringIO()):
                self.ctrl.open_hl_log()
                t0 = time.perf_counter()
                for _ in range(250):
                    self.assertIsNone(self.ctrl.wbc_step())
                dt = (time.perf_counter() - t0) / 250
            self.ctrl.close_hl_log()
            (name,) = os.listdir(tmp)
            with open(os.path.join(tmp, name)) as f:
                rows = [line.rstrip("\n").split(",") for line in f]
        self.assertEqual(rows[0], list(self.ctrl.HL_LOG_HEADER))
        self.assertEqual(len(rows), 1 + 50)
        self.assertTrue(all(len(r) == len(rows[0]) for r in rows))
        print(f"\n[TIMING] mean wbc_step (HL every 5th) = {dt * 1e3:.3f} ms of a 20 ms WBC period")
        self.assertLess(dt, 0.010)

    def test_mocap_and_fk_disk_agree_when_calibrated(self):
        self.ctrl.read_perception()
        np.testing.assert_allclose(self.ctrl.last_puck_d, [0.0, 0.0, 0.0425], atol=1e-5)
        np.testing.assert_allclose(self.ctrl.last_puck_d_mocap, self.ctrl.last_puck_d_fk, atol=1e-5)
        dpos, drpy, angle = self.ctrl.last_disk_check
        self.assertLess(np.abs(dpos).max(), 1e-6)
        self.assertLess(angle, 1e-3)
        self.assertEqual(self.ctrl.disk_frame_source, "mocap")

    def test_mocap_disk_is_the_source_and_error_is_reported(self):
        self.perc.disk_err_pos_d = np.array([0.003, -0.002, 0.001])
        self.perc.disk_err_rpy = np.radians([0.0, 0.0, 2.0])
        self.ctrl.read_perception()
        dpos, drpy, angle = self.ctrl.last_disk_check
        np.testing.assert_allclose(dpos, [0.003, -0.002, 0.001], atol=1e-6)
        np.testing.assert_allclose(drpy, [0.0, 0.0, 2.0], atol=1e-4)
        self.assertAlmostEqual(angle, 2.0, places=4)
        # The actor sees the MOCAP disk: puck_D shifted by the injected error.
        R = q_to_mat(q_from_rpy(0.0, 0.0, np.radians(2.0)))
        expect = R.T @ (np.array([0.0, 0.0, 0.0425]) - [0.003, -0.002, 0.001])
        np.testing.assert_allclose(self.ctrl.last_puck_d, expect, atol=1e-5)
        np.testing.assert_allclose(self.ctrl.last_puck_d_fk, [0.0, 0.0, 0.0425], atol=1e-5)
        self.assertIn("disk mocap-FK", self.ctrl.disk_check_status())
        # Motive pivot fix = the inverse of the injected asset error, in asset axes.
        origin, R_fix = self.ctrl.last_asset_fix
        np.testing.assert_allclose(R_fix, R.T, atol=1e-5)
        np.testing.assert_allclose(origin, -R.T @ np.array([0.003, -0.002, 0.001]), atol=1e-6)
        self.assertIn("Motive disk pivot fix", self.ctrl.motive_fix_status())

    def test_motive_fix_detects_a_90_degree_asset(self):
        # Asset created with +Y_asset = -Z_D (the picture: green pointing down).
        self.perc.disk_err_rpy = np.radians([-90.0, 0.0, 0.0])
        self.ctrl.read_perception()
        _, R_fix = self.ctrl.last_asset_fix
        self.assertTrue(T.B2WZ1TransportMocapController.motive_fix_status(self.ctrl))
        np.testing.assert_allclose(R_fix[:, 2], [0.0, -1.0, 0.0], atol=1e-5)  # Z_D = -Y_asset
        self.assertIn("Z_D=-Y_asset", self.ctrl.motive_fix_status())

    def test_disk_loss_is_immediate(self):
        self.assertIsNone(self.steps(12))
        for reason in ("UNTRACKED", "STALE", "NOT_IN_STREAM"):
            self.perc.disk_valid, self.perc.disk_reason = False, reason
            self.assertIn(f"DISK:{reason}", self.steps(1))

    def test_fk_source_ignores_the_disk_body(self):
        ctrl, _, perc = make_static_controller(
            {"disk_frame_source": "fk", "mocap_disk_name": None, "mocap_disk_id": None})
        perc.disk_valid, perc.disk_reason = False, "NOT_IN_STREAM"
        perc.disk_err_pos_d = np.array([0.02, 0.0, 0.0])
        with redirect_stdout(io.StringIO()):
            ctrl.initialize_policy_state_and_history()
            for _ in range(20):
                self.assertIsNone(ctrl.wbc_step())
        np.testing.assert_allclose(ctrl.last_puck_d, [0.0, 0.0, 0.0425], atol=1e-5)

    def test_fk_source_with_optional_disk_asset(self):
        ctrl, _, perc = make_static_controller({"disk_frame_source": "fk"})
        self.assertEqual(T.validate_transport_setup(ctrl.cfg), [])
        perc.disk_err_pos_d = np.array([0.004, 0.0, 0.0])
        with redirect_stdout(io.StringIO()):
            ctrl.initialize_policy_state_and_history()
            self.assertIsNone(ctrl.wbc_step())
        # Observation = FK disk; the mocap disk still feeds the cross-check.
        np.testing.assert_allclose(ctrl.last_puck_d, [0.0, 0.0, 0.0425], atol=1e-5)
        np.testing.assert_allclose(ctrl.last_disk_check[0], [0.004, 0.0, 0.0], atol=1e-6)
        perc.disk_valid, perc.disk_reason = False, "UNTRACKED"
        with redirect_stdout(io.StringIO()):
            for _ in range(10):
                self.assertIsNone(ctrl.wbc_step())  # an untracked disk never faults in fk mode
        self.assertIsNone(ctrl.last_disk_check)

    def test_constructor_override_of_disk_frame_source(self):
        cfg = filled_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cfg(cfg, tmp)
            z1 = FakeZ1(cfg)
            disk = T.DiskGeometry(cfg["disk_center_offset_gripper"], cfg["disk_x_axis_gripper"],
                                  cfg["disk_y_axis_gripper"], cfg["disk_z_axis_gripper"],
                                  cfg["disk_thickness"], cfg["disk_radius"])
            with redirect_stdout(io.StringIO()):
                ctrl = T.B2WZ1TransportMocapController(path, z1=z1, perception=FakePerception(z1, disk),
                                                       connect_dds=False, disk_frame_source="fk")
        self.assertEqual(ctrl.disk_frame_source, "fk")

    def test_scene_state_for_visualizer(self):
        self.assertIsNone(self.steps(12))
        st = self.ctrl._scene_state(self.ctrl.last_snapshot, self.ctrl.b2w_joint_pos,
                                    self.ctrl.base_quat_wxyz, with_task=True)
        np.testing.assert_allclose(st["base_pos_w"], self.perc.p_root)
        pose = st["free_body_poses"][T.VIZ_PUCK_FREE_JOINT]
        np.testing.assert_allclose(pose["pos"], self.perc.get_latest_snapshot()["puck"]["position_world"])
        self.assertEqual(len(st["discs"]), 1)
        self.assertEqual(len(st["extra_frames"]), 2)  # asset triad + FK target triad
        self.assertEqual(st["ee_cmd_plb"].shape, (9,))
        # magenta EE reference = PLB kp0_ref rotated by the mocap yaw
        ref = [p for p in st["mocap_points"] if p["rgba"][:3] == [1.0, 0.0, 1.0]][0]["pos"]
        plb = np.array([self.perc.p_root[0], self.perc.p_root[1], 0.0])
        expect = plb + q_to_mat(self.perc.q_root) @ self.ctrl.task.ee_kp0_ref
        np.testing.assert_allclose(ref, expect, atol=1e-5)

    def test_arm_target_range_fault(self):
        self.ctrl.arm_target[:] = self.ctrl.arm_q_upper + 0.1
        self.assertIn("outside the live SDK range", self.ctrl.check_arm_target_range())

    def test_start_gate_blocks_initialization(self):
        ctrl, _, _ = make_static_controller(puck_d=(0.05, 0.0, 0.0425))
        with self.assertRaisesRegex(RuntimeError, "Not ready"):
            ctrl.initialize_policy_state_and_history()
        ctrl, _, perc = make_static_controller()
        perc.puck_valid, perc.puck_reason = False, "UNTRACKED"
        with self.assertRaisesRegex(RuntimeError, "PUCK:UNTRACKED"):
            ctrl.initialize_policy_state_and_history()

    def test_operator_dpad_edge_and_hold(self):
        rc = self.ctrl.remote_controller
        self.assertIsNone(self.steps(5 * 7))          # settle ends at HL 5
        rc.button[KeyMap.down] = 1                     # inside the 4-s min hold
        self.assertIsNone(self.steps(5 * 2))
        self.assertEqual(T.EE_TARGET_NAMES[self.ctrl.task.current_target], "HIGH")
        self.assertIsNone(self.steps(5 * 45))          # still held: no new edge
        self.assertEqual(T.EE_TARGET_NAMES[self.ctrl.task.current_target], "HIGH")
        rc.button[KeyMap.down] = 0
        self.assertIsNone(self.steps(1))
        rc.button[KeyMap.down] = 1                     # fresh press after the hold
        self.assertIsNone(self.steps(5 * 2))
        self.assertEqual(T.EE_TARGET_NAMES[self.ctrl.task.current_target], "LOW")

    def test_run_routes_faults_to_protection(self):
        ctrl = self.ctrl
        calls = {}
        noop = lambda *a, **k: None  # noqa: E731
        for name in ("print_banner", "wait_for_low_state", "read_robot_state", "zero_torque_state",
                     "move_arm_to_default", "hold_arm_default_until_A", "move_b2w_to_pose",
                     "hold_default_until_ready_and_A", "settle_on_runtime_gains",
                     "initialize_policy_state_and_history", "open_hl_log"):
            setattr(ctrl, name, noop)
        ctrl.z1.connect = noop
        ctrl.perception.initialize = noop
        ctrl.perception.start = noop
        ctrl.perception.stop = lambda: calls.setdefault("perception_stop", True)
        ctrl.low_cmd = None
        ctrl.enter_damping_protection_mode = lambda reason: calls.setdefault("protect", reason)
        ctrl.send_exit_damping_once = lambda: calls.setdefault("exit", True)
        with mock.patch("utils.command_helper.InitLowCmd", noop), redirect_stdout(io.StringIO()):
            ctrl.policy_loop = lambda: "PUCK DROP: test"
            ctrl.run()
            self.assertEqual(calls.pop("protect"), "PUCK DROP: test")
            ctrl.policy_loop = mock.Mock(side_effect=FloatingPointError("nan obs"))
            ctrl.run()
            self.assertIn("nan obs", calls.pop("protect"))
            ctrl.policy_loop = mock.Mock(side_effect=KeyboardInterrupt())
            ctrl.run()
            self.assertNotIn("protect", calls)
            self.assertTrue(calls["exit"])
            self.assertTrue(calls["perception_stop"])


# ======================================================================
# 6. Z1 forward kinematics (deploy env + Z1 SDK) vs the training MJCF
# ======================================================================


def _z1_sdk_available():
    if not os.path.isdir(Z1_SDK_LIB):
        return False, f"{Z1_SDK_LIB} missing"
    sys.path.insert(0, Z1_SDK_LIB)
    try:
        import unitree_arm_interface  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"unitree_arm_interface not importable here ({exc}); run in env_unitree_sdk2 with LD_LIBRARY_PATH"
    return True, ""


Z1_OK, Z1_SKIP = _z1_sdk_available()


@unittest.skipUnless(Z1_OK, Z1_SKIP)
class TestZ1ForwardKinematics(unittest.TestCase):
    """The deploy-side disk frame equals the training MJCF disk geom."""

    def test_fk_chain_and_disk_frame(self):
        import unitree_arm_interface as uai

        from utils.z1_helper import Z1ArmAdapter

        cfg = filled_cfg(z1_sdk_lib=Z1_SDK_LIB)
        with redirect_stdout(io.StringIO()):
            adapter = Z1ArmAdapter(cfg, REPO)
        adapter.arm_model = uai.Z1Model(np.zeros(3), 0.0, np.zeros(3), np.zeros((3, 3)))
        disk = T.DiskGeometry(cfg["disk_center_offset_gripper"], cfg["disk_x_axis_gripper"],
                              cfg["disk_y_axis_gripper"], cfg["disk_z_axis_gripper"],
                              cfg["disk_thickness"], cfg["disk_radius"])
        # MuJoCo cylinder axis is the geom's local z; disk +Z = stator +Y.
        pos_err, rot_err, disk_err, axis_err = [], [], [], []
        for s in load_fixture()["samples"]:
            adapter.q = np.asarray(s["q"], dtype=np.float32)
            p, R = adapter.compute_policy_ee_pose_in_base()
            pos_err.append(np.linalg.norm(p - s["stator_pos_b"]))
            rot_err.append(np.abs(R - np.asarray(s["stator_rot_b"])).max())
            c, Rd = disk.frame_in(p, R)
            disk_err.append(np.linalg.norm(c - s["disk_pos_b"]))
            axis_err.append(np.linalg.norm(Rd[:, 2] - np.asarray(s["disk_rot_b"])[:, 2]))
        print(f"\n[FK] max |stator pos| err {max(pos_err) * 1e3:.4f} mm, max |R| err {max(rot_err):.2e}, "
              f"disk centre {max(disk_err) * 1e3:.4f} mm, disk normal {max(axis_err):.2e}")
        self.assertLess(max(pos_err), 1e-4)
        self.assertLess(max(rot_err), 1e-4)
        self.assertLess(max(disk_err), 1e-4)
        self.assertLess(max(axis_err), 1e-4)


# ======================================================================
# 6a. Disk asset calibration tool (deploy/disk_mocap_calibration.py)
# ======================================================================


class TestDiskCalibration(unittest.TestCase):
    """Probe-marker fit of frame D in the asset frame, on a synthetic Motive scene."""

    def setUp(self):
        import disk_mocap_calibration as C

        self.C = C
        rng = np.random.default_rng(3)
        # True frame D in the world: arbitrary tilt/yaw, as on the robot.
        self.q_D = q_from_rpy(0.2, -0.15, 0.9)
        self.R_D = q_to_mat(self.q_D)
        self.c_D = np.array([1.2, -0.4, 1.05])
        # Motive asset: created level in a Y-up Motive (+Y_asset = disk
        # normal, +Z_asset = -X_D), pivot at the marker centroid 7 mm off,
        # plus a 3-deg mis-rotation.
        R_D_asset = q_to_mat(q_from_rpy(np.radians(90.0), 0.0, np.radians(-90.0))) @ \
            q_to_mat(q_from_rpy(0.0, np.radians(3.0), 0.0))
        self.R_asset = self.R_D @ R_D_asset
        self.p_asset = self.c_D + self.R_D @ np.array([0.004, -0.005, -0.003])
        self.h, self.t = 0.0075, 0.004
        top = self.c_D + self.R_D @ np.array([0.0, 0.0, self.t / 2 + self.h])
        self.probes_w = {"centre": top,
                         "+X": top + self.R_D @ np.array([0.08, 0.0, 0.0]),
                         "+Y": top + self.R_D @ np.array([0.0, 0.07, 0.0])}
        self.noise = rng.normal(scale=1e-4, size=(3, 3))

    def local(self, p_w):
        return self.R_asset.T @ (p_w - self.p_asset)

    def test_fit_recovers_frame_d_in_asset_coordinates(self):
        pts = {k: self.local(v) for k, v in self.probes_w.items()}
        origin, R = self.C.fit_disk_frame(pts["centre"], pts["+X"], pts["+Y"], self.h, self.t)
        np.testing.assert_allclose(origin, self.local(self.c_D), atol=1e-9)
        np.testing.assert_allclose(R, self.R_asset.T @ self.R_D, atol=1e-9)
        corr = self.C.describe_correction(origin, R)
        self.assertFalse(corr["pass"])
        self.assertTrue(corr["axis_map"]["Z_D"].startswith("+Y_asset"))
        # Applying the correction (pivot -> origin, axes -> R) makes the asset D.
        p_new = self.p_asset + self.R_asset @ origin
        R_new = self.R_asset @ R
        np.testing.assert_allclose(p_new, self.c_D, atol=1e-9)
        np.testing.assert_allclose(R_new, self.R_D, atol=1e-9)
        # ... and a re-probe of the corrected asset passes.
        pts2 = {k: R_new.T @ (v - p_new) for k, v in self.probes_w.items()}
        o2, R2 = self.C.fit_disk_frame(pts2["centre"], pts2["+X"], pts2["+Y"], self.h, self.t)
        self.assertTrue(self.C.describe_correction(o2, R2)["pass"])

    def test_fit_is_robust_to_probe_noise_and_rejects_bad_geometry(self):
        pts = {k: self.local(v) + n for (k, v), n in zip(self.probes_w.items(), self.noise)}
        origin, R = self.C.fit_disk_frame(pts["centre"], pts["+X"], pts["+Y"], self.h, self.t)
        self.assertLess(np.linalg.norm(origin - self.local(self.c_D)), 5e-4)
        _, ang = self.C.axis_angle_deg((self.R_asset.T @ self.R_D).T @ R)
        self.assertLess(ang, 0.5)
        with self.assertRaisesRegex(ValueError, "30 mm"):
            self.C.fit_disk_frame(pts["centre"], pts["centre"] + 0.01, pts["+Y"], self.h, self.t)
        with self.assertRaisesRegex(ValueError, "collinear"):
            self.C.fit_disk_frame(pts["centre"], pts["+X"], 2 * pts["+X"] - pts["centre"], self.h, self.t)

    def test_probe_selection_from_a_natnet_frame(self):
        rb = _rb(21, self.p_asset, quat_from_rotmat_wxyz(self.R_asset))
        frame = types.SimpleNamespace(frame_number=1, rigid_bodies={21: rb},
                                      unlabeled_markers=np.array([self.probes_w["+X"], [9.0, 9.0, 9.0]]))
        p, status = self.C.probe_in_asset(frame, 21, 0.25)
        self.assertEqual(status, "OK")
        np.testing.assert_allclose(p, self.local(self.probes_w["+X"]), atol=1e-6)  # float32 quat
        frame.unlabeled_markers = np.array([self.probes_w["+X"], self.probes_w["+Y"]])
        self.assertIsNone(self.C.probe_in_asset(frame, 21, 0.25)[0])  # two candidates
        frame.rigid_bodies = {21: _rb(21, self.p_asset, valid=False)}
        self.assertEqual(self.C.probe_in_asset(frame, 21, 0.25)[1], "disk UNTRACKED")


# ======================================================================
# 6b. MuJoCo visualizer on the transport scene MJCF (no window)
# ======================================================================


@unittest.skipUnless(HAVE_MUJOCO_VIEWER and os.path.isfile(MJ_SCENE_XML), "mujoco or scene XML missing")
class TestVisualizerScene(unittest.TestCase):
    def test_puck_free_body_and_extras(self):
        from utils.mj_visualizer import MujocoDebugVisualizer

        viz = MujocoDebugVisualizer(MJ_SCENE_XML, show_floor=False, show_light=False,
                                    free_joint_names=[T.VIZ_PUCK_FREE_JOINT])
        state = {"base_quat_wxyz": [1.0, 0.0, 0.0, 0.0], "base_pos_w": [0.2, 0.1, 0.6],
                 "base_height": 0.6, "b2w_joint_pos_policy": np.zeros(16), "z1_q": np.zeros(6),
                 "gripper_q_training": 0.0,
                 "free_body_poses": {T.VIZ_PUCK_FREE_JOINT: {"pos": [0.7, 0.0, 1.1],
                                                            "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}
        viz._apply_state(state)
        mujoco.mj_forward(viz._m, viz._d)
        bid = mujoco.mj_name2id(viz._m, mujoco.mjtObj.mjOBJ_BODY, "transport_puck")
        np.testing.assert_allclose(viz._d.xpos[bid], [0.7, 0.0, 1.1], atol=1e-9)
        state.pop("free_body_poses")
        viz._apply_state(state)
        mujoco.mj_forward(viz._m, viz._d)
        self.assertLess(viz._d.xpos[bid][2], -5.0)  # parked while untracked
        # Overlay drawing with discs / triads into a plain scene.
        scn = mujoco.MjvScene(viz._m, maxgeom=500)
        state.update(discs=[{"pos": [0.7, 0, 1.0], "mat": np.eye(3), "radius": 0.1}],
                     extra_frames=[{"pos": [0.7, 0, 1.0], "quat_wxyz": [1, 0, 0, 0]}],
                     ee_cmd_plb=np.zeros(9, dtype=np.float32))
        viz._draw_markers(types.SimpleNamespace(user_scn=scn), state)
        self.assertGreaterEqual(scn.ngeom, 1 + 3)


# ======================================================================
# 7. Sim2sim parity: the REAL controller code driving TransportSim
# ======================================================================


class SimZ1:
    """Z1 adapter view of a TransportSim: encoders + exact gripperStator FK."""

    def __init__(self, sim, cfg):
        self.sim = sim
        self.arm_kps_runtime = np.asarray(cfg["arm_kps_runtime"], dtype=np.float32)
        self.arm_kds_runtime = np.asarray(cfg["arm_kds_runtime"], dtype=np.float32)
        self.unitree_arm_interface = types.SimpleNamespace(
            ArmFSMState=types.SimpleNamespace(PASSIVE="PASSIVE"))
        self.gripper_q = 0.0
        self.read_state()

    def read_state(self):
        d, nj = self.sim.d, self.sim.nj
        self.q = d.qpos[7:7 + nj][self.sim.arm_idx].astype(np.float32)
        self.qd = d.qvel[6:6 + nj][self.sim.arm_idx].astype(np.float32)

    def get_fsm_state(self):
        return "LOWCMD"

    def compute_policy_ee_pose_in_base(self):
        d, s = self.sim.d, self.sim
        Rb, pb = d.xmat[s.base_bid].reshape(3, 3), d.xpos[s.base_bid]
        return ((Rb.T @ (d.xpos[s.ee_bid] - pb)).astype(np.float32),
                (Rb.T @ d.xmat[s.ee_bid].reshape(3, 3)).astype(np.float32))


class SimPerception:
    """OptiTrack view of a TransportSim: base_link pose + puck centre."""

    def __init__(self, sim):
        self.sim = sim

    def get_latest_snapshot(self):
        d = self.sim.d
        p_root, q_root = d.qpos[0:3].copy(), d.qpos[3:7].copy()
        puck_w = d.xpos[self.sim.puck_bid].copy()
        puck_b = q_to_mat(q_root).T @ (puck_w - p_root)
        # Ideal mocap disk asset = the training disk frame of the sim.
        c_w, _, _, _, q_w_d = self.sim.get_disk_frame_world()
        return {"body": {"valid": True, "age_ms": 0.0, "root_position_world": p_root.tolist(),
                         "root_quat_wxyz": q_root.tolist()},
                "base_height": {"valid": True, "height_m": float(p_root[2]) - 0.0, "reason": "OK"},
                "puck": {"valid": True, "reason": "OK", "age_ms": 0.0, "position_base": puck_b.tolist(),
                         "position_world": puck_w.tolist(), "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
                "disk": {"valid": True, "reason": "OK", "age_ms": 0.0,
                         "position_world": np.asarray(c_w, dtype=np.float64).tolist(),
                         "quat_wxyz": np.asarray(q_w_d, dtype=np.float64).tolist()}}


def sim_low_state(sim, hardware_joint_names):
    d, m = sim.d, sim.m
    names = sim.MUJOCO_JOINT_NAMES
    motors = [types.SimpleNamespace(q=float(d.qpos[7 + names.index(n)]), dq=float(d.qvel[6 + names.index(n)]))
              for n in hardware_joint_names]
    return types.SimpleNamespace(
        tick=1,
        imu_state=types.SimpleNamespace(quaternion=d.qpos[3:7].tolist(),
                                        gyroscope=get_sensor_slice(m, d, "imu_gyro").tolist()),
        motor_state=motors)


@unittest.skipUnless(HAVE_SIM, SIM_SKIP)
class TestSim2SimParity(unittest.TestCase):
    SEED, MASS, MU, DURATION_S = 3, 0.5, 0.6, 9.0

    @classmethod
    def setUpClass(cls):
        with open(MJ_CFG) as f:
            cls.mj_cfg = yaml.safe_load(f)
        cls.cfg = filled_cfg()
        cls.tmp = tempfile.mkdtemp(prefix="transport_parity_")
        cls.cfg_path = write_cfg(cls.cfg, cls.tmp)
        cls.high_path = os.path.join(REPO, cls.cfg["high_policy_path"])

    def new_sim(self):
        with redirect_stdout(io.StringIO()):
            sim = TransportSim(self.mj_cfg)
        horizon = int(round(self.DURATION_S / 0.1)) + 10
        scenario = deploy_rng_scenario(self.SEED, self.mj_cfg, horizon, self.MASS, self.MU)
        return sim, scenario

    def test_keypoints_and_decode_match_sim(self):
        sim, scenario = self.new_sim()
        sim.reset(scenario)
        rng = np.random.default_rng(7)
        task = T.TransportTask(self.cfg, 0.1)
        worst = 0.0
        for i in range(1000):
            kp0 = rng.uniform([0.3, -0.5, 0.0], [1.0, 0.5, 1.3]).astype(np.float32)
            rot = rng.uniform([-1.2, -0.9, 0.6], [1.2, 0.9, 2.5]).astype(np.float32)
            a = rng.uniform(-1.5, 1.5, size=6).astype(np.float32)
            settling = i % 5 == 0
            sim.ee_kp0_ref, (sim.ee_yaw_ref, sim.ee_pitch_ref, sim.ee_roll_ref) = kp0.copy(), map(float, rot)
            sim.settle_steps_remaining = 1 if settling else 0
            raw, eff, kp0_cmd, rot_cmd, executed, _ = decode_residual("full", a, sim)
            sim.set_ee_command(kp0_cmd, rot_cmd)
            task.ee_kp0_ref = kp0.copy()
            task.ee_yaw_ref, task.ee_pitch_ref, task.ee_roll_ref = map(float, rot)
            task.settle_steps_remaining = 1 if settling else 0
            task.decode(a)
            for mine, ref in ((task.raw_action, raw), (task.effective_action, eff),
                              (task.kp0_cmd, kp0_cmd), (task.rot_cmd, rot_cmd),
                              (task.executed_action, executed), (task.ee_cmd_plb, sim.ee_cmd_plb)):
                worst = max(worst, float(np.abs(np.asarray(mine) - np.asarray(ref)).max()))
            np.testing.assert_array_equal(
                T.build_keypoints_from_kp0_yaw_pitch_roll_plb(kp0, *rot),
                sim_keypoints(kp0, *map(float, rot)))
        print(f"\n[ACTION] 1000 random decodes: max |deploy - sim2sim| = {worst:.3e}")
        self.assertEqual(worst, 0.0)

    def run_closed_loop(self, seed, mass, mu, duration_s):
        """wbc_step() of the deploy controller closes the loop on TransportSim.

        At every HL boundary the reference (HLMethod 'full' + the TransportSim
        reference process, both bit-exact to the sim2sim script) runs on the
        SAME physical state; the operator sources replay the sim's sampled
        references. Compared every step: actor history (192-D), raw/executed
        residual, kp0 reference, EE command, base command, WBC history
        (400-D), WBC action. Physics is driven by the CONTROLLER's targets.

        mj_forward() after each physics block makes xpos/xmat/sensors describe
        the post-step qpos, as one hardware sample does. (Plain mj_step leaves
        them one 4-ms physics step behind qpos; the sim2sim script mixes the
        two, the robot cannot.)
        """
        with redirect_stdout(io.StringIO()):
            sim = TransportSim(self.mj_cfg)
        scenario = deploy_rng_scenario(seed, self.mj_cfg, int(round(duration_s / 0.1)) + 10, mass, mu)
        sim.reset(scenario)
        method = HLMethod("full", self.high_path)
        method.reset(sim)

        z1, perc = SimZ1(sim, self.cfg), SimPerception(sim)
        with redirect_stdout(io.StringIO()):
            ctrl = T.B2WZ1TransportMocapController(self.cfg_path, z1=z1, perception=perc, connect_dds=False)
        ctrl.remote_controller = RemoteController()
        ctrl.read_operator_base_reference = lambda: sim.base_velocity_ref.copy()
        ctrl.poll_operator_ee_request = lambda: None
        ctrl.low_state = sim_low_state(sim, ctrl.hardware_joint_names)
        with redirect_stdout(io.StringIO()):
            ctrl.initialize_policy_state_and_history()

        err = {k: 0.0 for k in ("hl_obs", "raw", "executed", "kp0_ref", "ee_cmd", "ll_obs", "ll_action")}
        proxy_vs_force = []
        sim_drop_hl, end_reason = None, "duration"
        for _ in range(int(round(duration_s / 0.02))):
            if ctrl.ll_tick % 5 == 0:
                if ctrl.hl_tick > 0:
                    chk = sim.post_step_checks()
                    puck_d = sim.get_puck_pos_disk()
                    if not sim.settling:
                        proxy_vs_force.append((ctrl.contact_monitor.in_contact(puck_d), chk["puck_disk_contact"],
                                               float(np.hypot(puck_d[0], puck_d[1]))))
                    if chk["puck_drop"] and sim_drop_hl is None:
                        sim_drop_hl = ctrl.hl_tick
                    ee_i = sim._ee_i
                    sim.advance_references()
                    if sim._ee_i != ee_i:
                        ctrl.task.request_target(sim.current_ee_pose_index)
                out = method.step(sim)
            ctrl.low_state = sim_low_state(sim, ctrl.hardware_joint_names)
            with redirect_stdout(io.StringIO()):
                reason = ctrl.wbc_step()
            if reason is not None:
                end_reason = reason
                break
            if (ctrl.ll_tick - 1) % 5 == 0:
                err["hl_obs"] = max(err["hl_obs"], float(np.abs(ctrl.hl_history.flat() - method.history.flat()).max()))
                err["raw"] = max(err["raw"], float(np.abs(ctrl.task.raw_action - out["raw"]).max()))
                err["executed"] = max(err["executed"], float(np.abs(ctrl.task.executed_action - out["executed"]).max()))
                err["kp0_ref"] = max(err["kp0_ref"], float(np.abs(ctrl.task.ee_kp0_ref - sim.ee_kp0_ref).max()))
                err["ee_cmd"] = max(err["ee_cmd"], float(np.abs(ctrl.task.ee_cmd_plb - sim.ee_cmd_plb).max()))
                self.assertEqual(ctrl.task.current_target, sim.current_ee_pose_index)
                np.testing.assert_array_equal(ctrl.task.base_command, sim.base_command)
            sim.run_low_level_policy()
            err["ll_obs"] = max(err["ll_obs"], float(np.abs(ctrl.ll_history.flat() - sim.ll_history.flat()).max()))
            err["ll_action"] = max(err["ll_action"], float(np.abs(ctrl.last_ll_action - sim.last_ll_action).max()))
            sim.leg_target, sim.arm_target, sim.wheel_cmd = (
                ctrl.leg_target.copy(), ctrl.arm_target.copy(), ctrl.wheel_cmd.copy())
            for _ in range(sim.ll_decimation):
                sim.apply_low_level_actuation()
                mujoco.mj_step(sim.m, sim.d)
            mujoco.mj_forward(sim.m, sim.d)

        stats = {
            "hl_steps": ctrl.hl_tick, "end": end_reason, "sim_drop_hl": sim_drop_hl, "err": err,
            "agree": float(np.mean([p == f for p, f, _ in proxy_vs_force])) if proxy_vs_force else float("nan"),
            "proxy_lost_while_force_contact": sum(1 for p, f, _ in proxy_vs_force if f and not p),
            "proxy_lost_over_disk": sum(1 for p, f, rho in proxy_vs_force
                                        if f and not p and rho <= ctrl.disk.radius),
            "proxy_contact_while_force_lost": sum(1 for p, f, _ in proxy_vs_force if p and not f),
            "final_puck_d": sim.get_puck_pos_disk(),
            "n_active": len(proxy_vs_force),
        }
        print(f"\n[PARITY seed={seed} m={mass} mu={mu}] {stats['hl_steps']} HL steps; end: {end_reason}; "
              f"sim force-contact puck_drop at HL {sim_drop_hl}")
        print("[PARITY] max |deploy - sim2sim|: " + ", ".join(f"{k}={v:.2e}" for k, v in err.items()))
        print(f"[PARITY] contact proxy vs force: agreement {stats['agree']:.3f} over {stats['n_active']} "
              f"active HL steps | proxy lost/force contact: {stats['proxy_lost_while_force_contact']} | "
              f"proxy contact/force lost: {stats['proxy_contact_while_force_lost']}")
        for k, v in err.items():
            self.assertLess(v, 1e-4, k)
        # The proxy may only call "lost" before the force detector when the
        # puck centre is already past the rim (tipping over the edge).
        self.assertEqual(stats["proxy_lost_over_disk"], 0)
        return stats

    def test_closed_loop_puck_lost_low_friction(self):
        """mu=0.6 (sim2sim default): the puck is lost; the proxy must catch it."""
        st = self.run_closed_loop(seed=3, mass=0.5, mu=0.6, duration_s=12.0)
        self.assertTrue(st["end"].startswith("PUCK DROP"), st["end"])
        d = st["final_puck_d"]
        self.assertTrue(np.hypot(d[0], d[1]) > 0.1 or d[2] < 0.0, f"puck not actually off the disk: {d}")

    def test_contact_proxy_vs_training_detector(self):
        """Pure sim2sim episodes: geometric proxy drop vs force-based puck_drop."""
        with redirect_stdout(io.StringIO()):
            sim = TransportSim(self.mj_cfg)
        rows = []
        for seed in (1, 2, 3, 4):
            for mu in (0.4, 0.6):
                sim.reset(deploy_rng_scenario(seed, self.mj_cfg, 160, 0.5, mu))
                method = HLMethod("full", self.high_path)
                method.reset(sim)
                mon = T.PuckContactMonitor(0.1, 0.042, 0.01, 5)
                force_drop = proxy_drop = None
                for k in range(1, 150):
                    method.step(sim)
                    sim.run_hl_interval()
                    chk = sim.post_step_checks()
                    _, _, drop = mon.update(sim.get_puck_pos_disk(), sim.settling)
                    if chk["puck_drop"] and force_drop is None:
                        force_drop = k
                    if drop and proxy_drop is None:
                        proxy_drop = k
                    if force_drop and proxy_drop:
                        break
                    if (force_drop or proxy_drop) and k > max(force_drop or 0, proxy_drop or 0) + 5:
                        break
                    sim.advance_references()
                rows.append((seed, mu, force_drop, proxy_drop))
        print("\n[CONTACT] (seed, mu, training force-drop HL, proxy-drop HL): " + str(rows))
        for seed, mu, f, p in rows:
            with self.subTest(seed=seed, mu=mu):
                self.assertEqual(f is None, p is None)
                if f is not None:
                    self.assertLessEqual(abs(f - p), 2)

    def test_closed_loop_high_friction(self):
        st = self.run_closed_loop(seed=3, mass=0.5, mu=1.0, duration_s=12.0)
        self.assertGreaterEqual(st["hl_steps"], 60)
        if st["sim_drop_hl"] is None:
            self.assertEqual(st["end"], "duration")

if __name__ == "__main__":
    unittest.main(verbosity=2)
