"""
B2WZ1 filming-demo sim2real deployment: replay recorded MuJoCo commands from .npz.

Put this file next to b2wz1_locomanipulation_plb.py:
    deploy/b2wz1_filming.py

Run:
    python3 deploy/b2wz1_filming.py <network_interface> deploy/configs/b2wz1_filming.yaml --mode full-policy

Replay behavior:
  - base_command is replayed from .npz
  - ee_cmd_plb is replayed from .npz
  - replay finishes after replay_duration_s, then automatically enters joystick takeover

Safety:
  - During replay, press joystick_takeover_button, default A, to interrupt and enter takeover.
  - In takeover, joystick controls base velocity, policy controls legs/wheels, arm policy is ignored.
  - In takeover, Z1 arm smoothly moves to takeover_arm_target, usually joint zero.
  - Press SELECT at any time in the main loop to enter inherited high-KD damping mode.

Expected .npz keys:
  - base_cmds / base_commands / base_cmd / base_command: (N, 3)
  - ee_cmds_plb / ee_cmds / ee_commands_plb / ee_commands / ee_cmd_plb / ee_cmd: (N, 9)
  - optional control_dt scalar
  - optional t / time_s / timestamps / times: (N,)
"""

import os
import sys
import time
import argparse
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from b2wz1_locomanipulation_plb import B2WZ1PLBLocoManipController
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from utils.remote_controller import KeyMap


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 3.0 * t * t - 2.0 * t * t * t


def resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_str))


class NPZFilmingReplayCommandSampler:
    BASE_KEYS = ["base_cmds", "base_commands", "base_cmd", "base_command"]
    EE_KEYS = ["ee_cmds_plb", "ee_cmds", "ee_commands_plb", "ee_commands", "ee_cmd_plb", "ee_cmd"]
    TIME_KEYS = ["t", "time_s", "timestamps", "times"]

    def __init__(
        self,
        controller,
        replay_path: str,
        control_dt: float,
        replay_duration_s: Optional[float] = None,
        loop: bool = False,
        hold_last_when_done: bool = True,
    ):
        self.controller = controller
        self.replay_path = resolve_path(replay_path)
        self.control_dt = float(control_dt)
        self.replay_duration_s = None if replay_duration_s is None else float(replay_duration_s)
        self.loop = bool(loop)
        self.hold_last_when_done = bool(hold_last_when_done)

        if not os.path.exists(self.replay_path):
            raise FileNotFoundError(f"Replay .npz not found: {self.replay_path}")

        data = np.load(self.replay_path, allow_pickle=True)

        base_key = self._find_key(data, self.BASE_KEYS)
        ee_key = self._find_key(data, self.EE_KEYS)
        if base_key is None:
            raise KeyError(f"Replay file missing base command key. Tried: {self.BASE_KEYS}")
        if ee_key is None:
            raise KeyError(f"Replay file missing EE command key. Tried: {self.EE_KEYS}")

        self.base_cmds = np.asarray(data[base_key], dtype=np.float32)
        self.ee_cmds_plb = np.asarray(data[ee_key], dtype=np.float32)

        if self.base_cmds.ndim != 2 or self.base_cmds.shape[1] != 3:
            raise ValueError(f"Expected base commands shape (N, 3), got {self.base_cmds.shape}")
        if self.ee_cmds_plb.ndim != 2 or self.ee_cmds_plb.shape[1] != 9:
            raise ValueError(f"Expected EE commands shape (N, 9), got {self.ee_cmds_plb.shape}")
        if self.base_cmds.shape[0] != self.ee_cmds_plb.shape[0]:
            raise ValueError(
                f"Replay length mismatch: base has {self.base_cmds.shape[0]}, "
                f"EE has {self.ee_cmds_plb.shape[0]}"
            )

        self.num_steps_total = int(self.base_cmds.shape[0])

        self.timestamps = None
        time_key = self._find_key(data, self.TIME_KEYS)
        if time_key is not None:
            ts = np.asarray(data[time_key], dtype=np.float32).reshape(-1)
            if ts.shape[0] == self.num_steps_total:
                self.timestamps = ts
            else:
                print(f"[REPLAY][WARN] ignoring timestamp key {time_key}: shape={ts.shape}")

        file_control_dt = None
        if "control_dt" in data.files:
            try:
                file_control_dt = float(np.asarray(data["control_dt"]).reshape(-1)[0])
            except Exception:
                file_control_dt = None

        if file_control_dt is not None and abs(file_control_dt - self.control_dt) > 1e-6:
            print(
                f"[REPLAY][WARN] file control_dt={file_control_dt:.6f}, "
                f"controller control_dt={self.control_dt:.6f}; replaying one row per controller step."
            )

        if self.replay_duration_s is not None and self.replay_duration_s > 0.0:
            self.num_steps_replay = min(
                self.num_steps_total,
                max(1, int(round(self.replay_duration_s / self.control_dt))),
            )
        else:
            self.num_steps_replay = self.num_steps_total

        self.keypoints_command_plb = np.zeros(9, dtype=np.float32)
        self._step_in_cycle = 0
        self._done = False
        self._has_reset = False

        print("=" * 80)
        print("[REPLAY] Loaded filming replay")
        print(f"[REPLAY] path            : {self.replay_path}")
        print(f"[REPLAY] base key        : {base_key}, shape={self.base_cmds.shape}")
        print(f"[REPLAY] ee key          : {ee_key}, shape={self.ee_cmds_plb.shape}")
        print(f"[REPLAY] total rows      : {self.num_steps_total}")
        print(f"[REPLAY] replay rows     : {self.num_steps_replay}")
        print(f"[REPLAY] control_dt      : {self.control_dt:.4f}s")
        print(f"[REPLAY] replay duration : {self.num_steps_replay * self.control_dt:.2f}s")
        print(f"[REPLAY] loop            : {self.loop}")
        print("=" * 80)

    @staticmethod
    def _find_key(npz_data, candidates):
        keys = set(npz_data.files)
        for key in candidates:
            if key in keys:
                return key
        return None

    @property
    def command(self) -> np.ndarray:
        return self.keypoints_command_plb.copy()

    @property
    def cycle_duration_s(self) -> float:
        return float(self.num_steps_replay * self.control_dt)

    @property
    def step_in_cycle(self) -> int:
        return int(self._step_in_cycle)

    @property
    def done(self) -> bool:
        return bool(self._done)

    def reset(self, initial_kps_plb: np.ndarray, sample_first: bool = True):
        self._step_in_cycle = 0
        self._done = False
        self._has_reset = True

        self.controller.base_command[:] = self.base_cmds[0].copy()
        self.keypoints_command_plb = self.ee_cmds_plb[0].copy().astype(np.float32)

        print("[REPLAY] Reset to row 0.")
        print(f"[REPLAY] initial base_cmd={np.round(self.controller.base_command, 4)}")
        print(f"[REPLAY] initial kp0_plb={np.round(self.keypoints_command_plb[0:3], 4)}")

    def update(self) -> np.ndarray:
        if not self._has_reset:
            raise RuntimeError("Replay sampler not initialized. Call reset() first.")

        if self._done:
            if self.hold_last_when_done:
                idx = self.num_steps_replay - 1
                self.controller.base_command[:] = self.base_cmds[idx].copy()
                self.keypoints_command_plb = self.ee_cmds_plb[idx].copy().astype(np.float32)
            else:
                self.controller.base_command[:] = 0.0
            return self.command

        idx = int(self._step_in_cycle)

        if idx >= self.num_steps_replay:
            if self.loop:
                idx = idx % self.num_steps_replay
                self._step_in_cycle = idx
            else:
                self._done = True
                idx = self.num_steps_replay - 1

        self.controller.base_command[:] = self.base_cmds[idx].copy()
        self.keypoints_command_plb = self.ee_cmds_plb[idx].copy().astype(np.float32)

        self._step_in_cycle += 1
        if (not self.loop) and self._step_in_cycle >= self.num_steps_replay:
            self._done = True

        return self.command


class B2WZ1FilmingReplayPLBController(B2WZ1PLBLocoManipController):
    def __init__(self, cfg_path: str, mode: str):
        super().__init__(cfg_path=cfg_path, mode=mode)

        self.replay_path = self.cfg.get(
            "replay_npz_path",
            self.cfg.get("filming_replay_npz_path", "utils/dataset/filming_command_replay.npz"),
        )
        self.replay_duration_s = float(self.cfg.get("replay_duration_s", 15.0))
        self.replay_loop = bool(self.cfg.get("replay_loop", False))
        self.replay_hold_last_when_done = bool(self.cfg.get("replay_hold_last_when_done", True))

        self.ee_cmd_sampler = NPZFilmingReplayCommandSampler(
            controller=self,
            replay_path=self.replay_path,
            control_dt=self.control_dt,
            replay_duration_s=self.replay_duration_s,
            loop=self.replay_loop,
            hold_last_when_done=self.replay_hold_last_when_done,
        )

        self.replay_active = True

        self.joystick_takeover_active = False
        self.joystick_takeover_button_name = str(self.cfg.get("joystick_takeover_button", "A"))
        self.takeover_interrupt_enable_delay_s = float(self.cfg.get("takeover_interrupt_enable_delay_s", 0.8))
        self.main_loop_start_time = None

        self.takeover_arm_target = np.array(
            self.cfg.get("takeover_arm_target", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.takeover_arm_transition_s = float(self.cfg.get("takeover_arm_transition_s", 4.0))
        self.takeover_hold_ee_current = bool(self.cfg.get("takeover_hold_ee_current", True))
        self.takeover_use_startup_arm_gains = bool(self.cfg.get("takeover_use_startup_arm_gains", True))
        self.takeover_start_time = 0.0
        self.takeover_tick = 0
        self.takeover_arm_start = self.default_arm_pos.copy()
        self.takeover_ee_cmd_plb = np.zeros(9, dtype=np.float32)

        print("=" * 80)
        print("[B2WZ1-FILM-REPLAY] Controller initialized")
        print(f"[B2WZ1-FILM-REPLAY] replay_path       : {self.replay_path}")
        print(f"[B2WZ1-FILM-REPLAY] replay_duration_s : {self.replay_duration_s:.2f}")
        print(f"[B2WZ1-FILM-REPLAY] interrupt button  : {self.joystick_takeover_button_name}")
        print(f"[B2WZ1-FILM-REPLAY] interrupt delay   : {self.takeover_interrupt_enable_delay_s:.2f}s")
        print(f"[B2WZ1-FILM-REPLAY] takeover arm tgt  : {np.round(self.takeover_arm_target, 3)}")
        print("=" * 80)

    def _update_base_command_from_remote(self):
        """Disable inherited joystick command update during replay."""
        return

    def setup(self):
        super().setup()
        print("=" * 80)
        print("B2WZ1 filming replay settings")
        print(f"Replay path       : {resolve_path(self.replay_path)}")
        print(f"Replay duration   : {self.ee_cmd_sampler.cycle_duration_s:.2f}s")
        print("Base command mode : replay .npz during filming; joystick during takeover")
        print("EE command mode   : replay .npz during filming; hold current during takeover")
        print(f"Interrupt         : press {self.joystick_takeover_button_name} to enter joystick lock-arm takeover")
        print(f"Interrupt delay   : {self.takeover_interrupt_enable_delay_s:.2f}s after replay starts")
        print("Protection        : press SELECT for high-KD damping")
        print("=" * 80)

    def _button_pressed_by_name(self, name: str) -> bool:
        if name is None:
            return False
        name = str(name).strip()
        if not name:
            return False

        try:
            key = int(name)
        except ValueError:
            if hasattr(KeyMap, name):
                key = getattr(KeyMap, name)
            elif hasattr(KeyMap, name.lower()):
                key = getattr(KeyMap, name.lower())
            elif hasattr(KeyMap, name.upper()):
                key = getattr(KeyMap, name.upper())
            else:
                print(f"[B2WZ1-FILM-REPLAY][WARN] Unknown button name: {name}")
                return False

        try:
            return self.remote_controller.button[key] == 1
        except Exception:
            return False

    def _takeover_interrupt_enabled(self) -> bool:
        if self.main_loop_start_time is None:
            self.main_loop_start_time = time.perf_counter()
            return False
        return (time.perf_counter() - self.main_loop_start_time) >= self.takeover_interrupt_enable_delay_s

    def _update_takeover_base_command_from_remote(self):
        cmd = np.zeros(3, dtype=np.float32)

        cmd[0] = np.clip(self.remote_controller.ly, -1.0, 1.0) * self.command_scale[0]
        cmd[1] = np.clip(-self.remote_controller.lx, -1.0, 1.0) * self.command_scale[1]
        cmd[2] = np.clip(-self.remote_controller.rx, -1.0, 1.0) * self.command_scale[2]

        lin_norm = np.linalg.norm(cmd[:2], ord=2)
        if lin_norm < self.command_deadband_lin * max(self.command_scale[0], self.command_scale[1]):
            cmd[0] = 0.0
            cmd[1] = 0.0
        if abs(cmd[2]) < self.command_deadband_ang * self.command_scale[2]:
            cmd[2] = 0.0

        self.base_command[:] = cmd

    def enter_joystick_takeover(self, reason: str = "manual"):
        if self.joystick_takeover_active:
            return

        self._read_all_sensors_once()

        self.replay_active = False
        self.joystick_takeover_active = True
        self.takeover_start_time = time.perf_counter()
        self.takeover_tick = 0
        self.takeover_arm_start = self.z1.q.copy().astype(np.float32)

        if self.takeover_hold_ee_current:
            try:
                self.takeover_ee_cmd_plb = self.compute_ee_current_kp_plb().copy().astype(np.float32)
            except Exception:
                self.takeover_ee_cmd_plb = self.ee_cmd_plb_current.copy().astype(np.float32)
        else:
            self.takeover_ee_cmd_plb = self.ee_cmd_plb_current.copy().astype(np.float32)

        self.last_action[:] = 0.0
        self.base_command[:] = 0.0
        self.ee_cmd_plb_current = self.takeover_ee_cmd_plb.copy()

        self.base_ang_vel_hist.clear()
        self.projected_gravity_hist.clear()
        self.base_cmd_hist.clear()
        self.ee_cmd_hist.clear()
        self.joint_pos_leg_hist.clear()
        self.joint_pos_arm_hist.clear()
        self.joint_vel_leg_hist.clear()
        self.joint_vel_arm_hist.clear()
        self.joint_vel_wheel_hist.clear()
        self.last_action_hist.clear()
        self._init_history()

        print("\n" + "=" * 80)
        print(f"[B2WZ1-FILM-REPLAY][TAKEOVER] Activated. reason={reason}")
        print("[B2WZ1-FILM-REPLAY][TAKEOVER] Mode logic: lock-arm-policy + joystick base command.")
        print(f"[B2WZ1-FILM-REPLAY][TAKEOVER] Arm start : {np.round(self.takeover_arm_start, 3)}")
        print(f"[B2WZ1-FILM-REPLAY][TAKEOVER] Arm target: {np.round(self.takeover_arm_target, 3)}")
        print(f"[B2WZ1-FILM-REPLAY][TAKEOVER] Arm transition: {self.takeover_arm_transition_s:.2f}s")
        print(f"[B2WZ1-FILM-REPLAY][TAKEOVER] Use startup arm gains: {self.takeover_use_startup_arm_gains}")
        print("[B2WZ1-FILM-REPLAY][TAKEOVER] Use joystick to drive base. Press SELECT for high damping.")
        print("=" * 80 + "\n")

    def _compute_takeover_arm_target(self) -> np.ndarray:
        elapsed = time.perf_counter() - self.takeover_start_time
        alpha = min(1.0, elapsed / max(1e-3, self.takeover_arm_transition_s))
        alpha = smoothstep(alpha)
        return ((1.0 - alpha) * self.takeover_arm_start + alpha * self.takeover_arm_target).astype(np.float32)

    def _append_obs_to_history(self, obs_step: np.ndarray):
        i = 0
        curr_base_ang_vel = obs_step[i:i + 3]; i += 3
        curr_projected_gravity = obs_step[i:i + 3]; i += 3
        curr_base_cmd = obs_step[i:i + 3]; i += 3
        curr_ee_cmd = obs_step[i:i + 9]; i += 9
        curr_joint_pos_leg = obs_step[i:i + 12]; i += 12
        curr_joint_pos_arm = obs_step[i:i + 6]; i += 6
        curr_joint_vel_leg = obs_step[i:i + 12]; i += 12
        curr_joint_vel_arm = obs_step[i:i + 6]; i += 6
        curr_joint_vel_wheel = obs_step[i:i + 4]; i += 4
        curr_last_action = obs_step[i:i + 22]; i += 22

        self.base_ang_vel_hist.append(curr_base_ang_vel.copy())
        self.projected_gravity_hist.append(curr_projected_gravity.copy())
        self.base_cmd_hist.append(curr_base_cmd.copy())
        self.ee_cmd_hist.append(curr_ee_cmd.copy())
        self.joint_pos_leg_hist.append(curr_joint_pos_leg.copy())
        self.joint_pos_arm_hist.append(curr_joint_pos_arm.copy())
        self.joint_vel_leg_hist.append(curr_joint_vel_leg.copy())
        self.joint_vel_arm_hist.append(curr_joint_vel_arm.copy())
        self.joint_vel_wheel_hist.append(curr_joint_vel_wheel.copy())
        self.last_action_hist.append(curr_last_action.copy())

    def _build_obs_stack_from_history(self) -> np.ndarray:
        obs_stack = np.concatenate(
            [
                np.array(self.base_ang_vel_hist).reshape(-1),
                np.array(self.projected_gravity_hist).reshape(-1),
                np.array(self.base_cmd_hist).reshape(-1),
                np.array(self.ee_cmd_hist).reshape(-1),
                np.array(self.joint_pos_leg_hist).reshape(-1),
                np.array(self.joint_pos_arm_hist).reshape(-1),
                np.array(self.joint_vel_leg_hist).reshape(-1),
                np.array(self.joint_vel_arm_hist).reshape(-1),
                np.array(self.joint_vel_wheel_hist).reshape(-1),
                np.array(self.last_action_hist).reshape(-1),
            ],
            dtype=np.float32,
        )
        assert obs_stack.shape[0] == self.obs_dim, f"obs_stack dim mismatch: {obs_stack.shape[0]} vs {self.obs_dim}"
        return obs_stack

    def _policy_forward(self, obs_stack: np.ndarray) -> np.ndarray:
        if self.mode == "pd-stand" or not self.use_policy:
            return np.zeros(self.action_dim, dtype=np.float32)
        return self.session.run([self.output_name], {self.input_name: obs_stack[None, :]})[0][0].astype(np.float32)

    def step(self):
        self._read_all_sensors_once()

        if self.remote_controller.button[KeyMap.select] == 1:
            return False

        if self.main_loop_start_time is None:
            self.main_loop_start_time = time.perf_counter()

        # ---------------------------------------------------------------------
        # Command source
        # ---------------------------------------------------------------------
        if self.joystick_takeover_active:
            self._update_takeover_base_command_from_remote()
            self.ee_cmd_plb_current = self.takeover_ee_cmd_plb.copy()
            active_phase = "TAKEOVER"

        else:
            if self._takeover_interrupt_enabled() and self._button_pressed_by_name(self.joystick_takeover_button_name):
                self.enter_joystick_takeover(reason=f"button_{self.joystick_takeover_button_name}")
                self._update_takeover_base_command_from_remote()
                self.ee_cmd_plb_current = self.takeover_ee_cmd_plb.copy()
                active_phase = "TAKEOVER"

            elif self.ee_cmd_sampler.done:
                # Replay finished on previous step; now enter takeover.
                # This ensures the last replay row was actually sent once.
                self.enter_joystick_takeover(reason="replay_finished")
                self._update_takeover_base_command_from_remote()
                self.ee_cmd_plb_current = self.takeover_ee_cmd_plb.copy()
                active_phase = "TAKEOVER"

            else:
                self.ee_cmd_plb_current = self.ee_cmd_sampler.update()
                active_phase = "REPLAY"

        # ---------------------------------------------------------------------
        # Observation and policy
        # ---------------------------------------------------------------------
        obs_step = self.build_obs_step(self.ee_cmd_plb_current)

        if self.debug_obs_enabled and self.debug_obs_started and self.debug_obs_print_count < self.debug_obs_print_max:
            self.print_obs_step_debug(obs_step, tag=f"control_step={self.policy_tick}, phase={active_phase}, mode={self.mode}")
            self.debug_obs_print_count += 1

        self._append_obs_to_history(obs_step)
        obs_stack = self._build_obs_stack_from_history()

        action = self._policy_forward(obs_stack)
        self.last_action[:] = action

        raw_leg_act = action[self.leg_action_indices].copy()
        raw_arm_act = action[self.arm_action_indices].copy()
        raw_wheel_act = action[self.wheel_action_indices].copy()

        # ---------------------------------------------------------------------
        # Action -> targets -> lowcmd
        # ---------------------------------------------------------------------
        if active_phase == "TAKEOVER":
            # Always lock arm during takeover. Legs/wheels use policy if available.
            if self.use_policy:
                self.leg_target = self.default_leg_pos_policy + self.leg_action_scale * raw_leg_act
                self.wheel_cmd[:] = self.wheel_action_scale * raw_wheel_act
            else:
                self.leg_target = self.default_leg_pos_policy.copy()
                self.wheel_cmd[:] = 0.0

            self.arm_target = self._compute_takeover_arm_target()

            self._write_b2w_rl_cmd(
                leg_target_policy=self.leg_target,
                wheel_cmd_policy=self.wheel_cmd,
            )
            self.send_b2w_cmd()

            self.z1.track_target_pd_once(
                q_target=self.arm_target.copy(),
                gripper_q_target=self.default_gripper_pos,
                use_startup_gains=self.takeover_use_startup_arm_gains,
            )
            self.takeover_tick += 1

        else:
            if self.mode == "pd-stand":
                self.leg_target = self.default_leg_pos_policy.copy()
                self.arm_target = self.default_arm_pos.copy()
                self.wheel_cmd[:] = 0.0
                self._write_b2w_pd_stand_cmd()

            elif self.mode == "lock-arm-policy":
                self.leg_target = self.default_leg_pos_policy + self.leg_action_scale * raw_leg_act
                self.arm_target = self.default_arm_pos.copy()
                self.wheel_cmd[:] = self.wheel_action_scale * raw_wheel_act
                self._write_b2w_rl_cmd(
                    leg_target_policy=self.leg_target,
                    wheel_cmd_policy=self.wheel_cmd,
                )

            elif self.mode == "full-policy":
                self.leg_target = self.default_leg_pos_policy + self.leg_action_scale * raw_leg_act
                self.wheel_cmd[:] = self.wheel_action_scale * raw_wheel_act

                if self.policy_tick < self.arm_enable_delay_steps:
                    self.arm_target = self.default_arm_pos.copy()
                else:
                    arm_target_policy = self.default_arm_pos + self.arm_action_scale * raw_arm_act
                    warmup_progress_steps = self.policy_tick - self.arm_enable_delay_steps
                    warmup_alpha = min(1.0, float(warmup_progress_steps) / float(self.arm_policy_warmup_steps))
                    self.arm_target = (
                        (1.0 - warmup_alpha) * self.default_arm_pos
                        + warmup_alpha * arm_target_policy
                    ).astype(np.float32)

                self._write_b2w_rl_cmd(
                    leg_target_policy=self.leg_target,
                    wheel_cmd_policy=self.wheel_cmd,
                )

            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            self.send_b2w_cmd()

            use_startup_gains = self.mode == "pd-stand"
            self.z1.track_target_pd_once(
                q_target=self.arm_target.copy(),
                gripper_q_target=self.default_gripper_pos,
                use_startup_gains=use_startup_gains,
            )

        self.counter += 1
        self.policy_tick += 1

        if self.counter % 100 == 0:
            leg_q_meas = self.b2w_joint_pos[:12].copy()
            wheel_dq_meas = self.b2w_joint_vel[12:16].copy()
            arm_q_meas = self.z1.q.copy()
            kp0 = self.ee_cmd_plb_current[0:3]

            if active_phase == "TAKEOVER":
                phase_extra = (
                    f"takeover_tick={self.takeover_tick} | "
                    f"arm_act_ignored=[{raw_arm_act.min():+.2f},{raw_arm_act.max():+.2f}]"
                )
            else:
                phase_extra = (
                    f"replay_step={self.ee_cmd_sampler.step_in_cycle}/{self.ee_cmd_sampler.num_steps_replay} | "
                    f"replay_done={self.ee_cmd_sampler.done}"
                )

            print(
                f"[{self.counter:5d}] {active_phase} | "
                f"mode={self.mode} | "
                f"cmd={np.round(self.base_command, 3)} | "
                f"kp0={np.round(kp0, 3)} | "
                f"{phase_extra} | "
                f"leg_act=[{raw_leg_act.min():+.2f},{raw_leg_act.max():+.2f}] | "
                f"wheel_act=[{raw_wheel_act.min():+.2f},{raw_wheel_act.max():+.2f}]"
            )
            print(
                "[POLICY-TARGET] "
                f"leg_q_tgt={np.round(self.leg_target, 3)} | "
                f"wheel_dq_tgt={np.round(self.wheel_cmd, 3)} | "
                f"arm_q_tgt={np.round(self.arm_target, 3)}"
            )
            print(
                "[JOINT-MEAS] "
                f"leg_q={np.round(leg_q_meas, 3)} | "
                f"wheel_dq={np.round(wheel_dq_meas, 3)} | "
                f"arm_q={np.round(arm_q_meas, 3)}"
            )

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="Network interface, e.g. enp3s0")
    parser.add_argument("config", type=str, help="Path to yaml config")
    parser.add_argument(
        "--mode",
        type=str,
        default="full-policy",
        choices=["pd-stand", "lock-arm-policy", "full-policy"],
        help="Run mode",
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)

    controller = B2WZ1FilmingReplayPLBController(cfg_path=args.config, mode=args.mode)
    controller.run()


if __name__ == "__main__":
    main()