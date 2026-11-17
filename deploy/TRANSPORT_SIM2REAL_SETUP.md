# B2W+Z1 Transport Sim2Real — Setup, Alignment and Preflight

Deployment: `deploy/b2wz1_hl_transport_uan_mocap.py` + `deploy/configs/b2wz1_hl_transport_uan_mocap.yaml`
Tests: `deploy/tests/test_b2wz1_hl_transport_uan_mocap.py`

Status as of 2026-11-16: **implemented and validated offline only.** The script has never run on
the robot. It refuses to start until every `[REQUIRED SETUP]` key in the YAML is filled
(see §F). Nothing below claims a physical step has been done.

---

## 0. What transport needs that retrieval does not

| | Retrieval sim2real | Transport sim2real |
|---|---|---|
| High-level policy | `high_policy_8.onnx`, 165→9, ABS commands | `b2wz1_hl_transport_uan/high_policy.onnx` (sha256 `db783046…`), 192→6, **residual** on an EE reference |
| Frozen WBC | `policy_wbc_uan_3.onnx` (`3b7874b7…`, 2026-09-07) | `b2wz1_wbc_uan_plb/policy_wbc_uan_transport.onnx` (`6b5aa5de…`, 2026-10-27). **Different checkpoint, never run on hardware.** |
| Mocap rigid bodies | B2, object (`cube4`), retrieval target | **B2 (reused) + puck (new).** No disk rigid body. |
| Gripper | binary open/close | held at the training position 0.0 (closed) for the whole run |
| Task references | none (policy chooses) | operator: sticks = base velocity, D-pad = EE target |
| End condition | operator / fault | operator SELECT, fault, or **puck drop** |

> **Do not use `exported_policies/b2wz1_hl_transport_uan/low_policy.onnx`** (`47f19872…`). It is the
> 2026-09-26 WBC of the superseded `high_policy_1` run. The YAML pins both policies by SHA-256 and
> the script refuses any other file. The correct WBC was copied byte-identically from
> `unitree_mujoco/exported_policy/b2wz1_wbc_plb_uan/policy_wbc_uan_transport.onnx`.

---

## 1. Observation alignment (actor frame, 64-D × 3 frames, feature-major = 192-D)

Reference: `unitree_mujoco/simulate_python/deploy_mujoco/b2wz1_hl_transport_uan.py::build_hl_obs_frame`.
Normalization is inside the ONNX graph; no inference-time noise (sim2sim play mode). History is
stacked feature-major, filled with the reset frame, and the first inference after reset uses the
filled history without appending (as in sim2sim). All HL features are sampled at the 10-Hz HL boundary.

| # | Feature | Dim | Sim2sim source | Real source | Frame | Units | Processing | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | base angular velocity | 3 | `imu_gyro` sensor | Unitree SDK `imu_state.gyroscope` | base | rad/s | none | Directly available (as retrieval/WBC) |
| 2 | projected gravity | 3 | root quat (qpos) | Unitree SDK `imu_state.quaternion` | base | unit | R(q)ᵀ·[0,0,−1] | Directly available |
| 3 | leg joint pos − default | 12 | qpos | Unitree SDK motor q (hw→policy order) | joint | rad | minus default | Directly available |
| 4 | arm joint pos − default | 6 | qpos | Z1 SDK q | joint | rad | minus default | Directly available |
| 5 | leg joint vel | 12 | qvel | Unitree SDK motor dq | joint | rad/s | none | Directly available |
| 6 | arm joint vel | 6 | qvel | Z1 SDK qd | joint | rad/s | none | Directly available |
| 7 | wheel joint vel | 4 | qvel | Unitree SDK motor dq | joint | rad/s | none | Directly available |
| 8 | base velocity reference | 3 | training sampler U(±0.6), held 3–5 s | **operator sticks** (b2wz1_wbc_uan.py mapping/deadband), clipped to U(±0.6) range, read at each HL step | base (vx,vy,wz) | m/s, rad/s | clip | Command history (operator) — see §2 |
| 9 | EE kp0 reference | 3 | HIGH (measured after settle) / LOW / LEFT / RIGHT + min-jerk | same process; **target chosen by D-pad**; HIGH measured by Z1 FK + IMU + mocap base height | PLB | m | min-jerk 1.5 s / 2.0 s (LOW) | Robot FK + mocap height + command history |
| 10 | EE orientation reference | 3 | constant canonical [0, −0.06, π/2] | same constant | PLB yaw/pitch/roll | rad | none | Constant (as training) |
| 11 | previous executed residual | 6 | `executed_hl_action` | same arithmetic (`TransportTask.decode`) | normalized | — | clip to [−1,1] after WBC bounds | Command history |
| 12 | **puck centre in disk frame** | 3 | MuJoCo puck body pos vs disk frame | **OptiTrack puck centre** (in mocap base_link frame) projected on the **Z1-FK disk frame** | disk D | m | R_Dᵀ(p_puck − c_D) | **OptiTrack + FK — requires puck rigid body + calibration (§A, §B)** |

PLB = yaw-only projected base frame at ground height (`ground_z`). On the robot PLB uses IMU
roll/pitch/yaw and the **mocap base height** — the same path the validated retrieval uses.

Feature 12 in detail (the only non-proprioceptive input):
```
puck_b   = R_root^T (p_puck_W + R_puck·mocap_puck_offset_local − p_root_W)     # MocapPerceptionSystem, mocap root = base_link
(p_s,R_s)= Z1ArmAdapter.compute_policy_ee_pose_in_base()                      # gripperStator in base frame
c_D      = p_s + R_s·[0.100, 0.045, 0.0]        R_D = R_s·[[1,0,0],[0,0,-1],[0,1,0]]ᵀ-columns   # disk +Z = stator +Y
puck_D   = R_Dᵀ (puck_b − c_D)                  # resting on the disk: z_D = 0.002 + 0.040 = 0.042 m
```
The projection is frame-invariant; mocap world yaw never enters (only the mocap base_link
orientation, whose roll/pitch agree with the IMU through the existing root calibration).

**No velocity observation exists for the puck**, so no velocity is estimated or differenced.
Training adds U(±0.01 m) noise to feature 12; this bounds how much measurement error the policy
has seen (see the error budget in §B).

The WBC (80-D × 5) observation is unchanged from the validated retrieval/WBC deployments: gyro,
gravity, base command, 9-D EE keypoints (PLB), leg/arm pos, leg/arm/wheel vel, last WBC action.

---

## 2. Action alignment (6-D residual)

Reference: sim2sim `decode_hl_action`; reproduced line-for-line in `TransportTask.decode`.

```
a_raw   = clip(policy_output, -1, 1)            # ONNX already outputs the Beta mean mapped to [-1,1]
a_eff   = 0 during the 0.5-s settle, else a_raw
kp0_cmd = clip(kp0_ref + a_eff[0:3]·0.10 m, x∈[0.40,0.90], y∈[-0.40,0.40], z∈[0.10,1.20])   (= kp0_ref during settle)
rot_cmd = clip([yaw,pitch,roll]_ref + a_eff[3:6]·π/4, yaw∈±60°, pitch∈±45°, roll∈[45°,135°])
exec    = clip([(kp0_cmd−kp0_ref)/0.10, (rot_cmd−rot_ref)/(π/4)], -1, 1)     → next obs feature 11
ee_cmd  = build_keypoints_from_kp0_yaw_pitch_roll_plb(kp0_cmd, yaw, pitch, roll, 0.30, 0.30)  → WBC (9-D, PLB)
base_command = base reference (no learned base residual)
```
* Position residual is additive in PLB; orientation residual is additive on [yaw, pitch, roll]
  (not a quaternion product). The keypoint builder is the **training adapter** (yaw/pitch define
  local +X, roll about +X) copied verbatim from sim2sim — **not** the retrieval ABS builder with its
  pitch sign flip.
* No smoothing, filtering or rate limiting anywhere (as sim2sim and the validated WBC loop).
* Timing: HL 10 Hz → WBC 50 Hz → hardware 500 Hz zero-order hold (identical to
  `b2wz1_wbc_uan.py` / retrieval). UAN is simulation-only; the Z1 runs firmware PD with audited gains.
* Reference process order per HL boundary (training/sim2sim): contact check on the completed
  transition → advance references (settle countdown / min-jerk step / operator inputs) → actor
  obs → inference → decode → WBC.

**Operator substitutions for the training reference sampler** (the only intentional semantic change):

| Training | Real | Why |
|---|---|---|
| base ref ~U(±0.6)³, piecewise constant 3–5 s | sticks, clipped to ±0.6, sampled each HL step, deadband 0.2 | a random base command on a real robot is unsafe; values stay inside the trained range, but continuous stick motion is smoother than training's step changes |
| EE target uniform over HIGH/LOW/LEFT/RIGHT every 4–5 s | D-pad up/down/left/right; rejected if < 4.0 s since last switch (the training minimum) | same set, same transitions (min-jerk 1.5 s, LOW 2.0 s); never faster than training |
| first targets sampled at settle end | base ref = sticks; EE target = pending D-pad request or HIGH | HIGH is a training target |
| episode timeout 15 s | none (operator stops with SELECT) | observations carry no time; training episodes were ≤15 s, so long runs extrapolate in duration only |

---

## A. OptiTrack rigid bodies

| Object | Rigid body? | Category | YAML keys | Used quantity | → observation |
|---|---|---|---|---|---|
| B2 base | **Yes** — existing asset `B2` (id 9) | **Reuse** from retrieval, unchanged | `mocap_body_name/id`, `mocap_root_frame_is_base_link: true` | base_link pose | base height (PLB, HIGH kp0, feature 9) and world→base transform for the puck (feature 12) |
| Puck | **Yes — new** | **New, required** | `mocap_puck_name`, `mocap_puck_id`, `mocap_puck_offset_local`, `mocap_puck_max_age_s: 0.10` | puck **centre position** only (orientation unused) | feature 12 |
| Disk on the gripper | **No** | Optional (one-time calibration aid only, no code path) | — | disk frame comes from Z1 FK + training mount | feature 12 |
| Retrieval target / cube | No | Not used | — | — | — |

Why no disk rigid body: the deploy FK chain reproduces the training MJCF disk to 0.0001 mm
(test `TestZ1ForwardKinematics`, 64 random arm poses), and the WBC itself already relies on this
FK. A disk rigid body would add markers right where the puck slides. Add one temporarily only if
the §E.4 disk-centre check fails and you need to separate FK/compliance error from calibration error.

**Puck rigid body — physical setup**
1. ≥ 4 markers in an **asymmetric** pattern on the **top face** of the puck (not the side: side
   markers would collide with nothing, but top markers stay visible from overhead cameras while the
   disk tilts). Keep marker bases inside the top face; nothing may protrude below the bottom face.
2. Weigh the puck **with markers**; that mass goes into `puck_mass_kg` and must be in [0.50, 1.50] kg.
3. Create the asset in Motive **with the puck standing upright on the floor**, so the asset frame's
   +Z is the cylinder axis (Motive aligns a new asset to the world axes).
4. Name it (e.g. `puck`) and note its streaming ID; it must differ from `B2` (id 9). Put the name in
   `mocap_puck_name` and the ID in `mocap_puck_id` (name is preferred; ID is the fallback).
5. Enable Rigid Bodies (and, for debugging, Labeled/Unlabeled Markers) in Motive streaming —
   same settings as retrieval.

---

## B. Calibration

| # | Calibration | Why | Observations affected | YAML | How to validate |
|---|---|---|---|---|---|
| B1 | OptiTrack world / ground plane | base height and PLB z are absolute | 9, 12 (via base frame), WBC EE command | `ground_z: 0.0` | **Reuse**: the retrieval setup. Re-run Motive ground-plane calibration if cameras moved. |
| B2 | B2 asset = base_link | every base-frame quantity | 9, 12 | `mocap_root_frame_is_base_link: true` (or `mocap_root_offset_path` from `deploy/b2w_mocap_root_calibration.py`) | **Reuse** from retrieval, but transport is far less tolerant: the disk centre is 0.86 m from base_link at HIGH, so a 1° roll/pitch/**yaw** error of the B2 asset vs base_link moves puck_D by ≈ 15 mm (more than the ±10 mm training noise). Retrieval only needed cm accuracy. If §E.4 shows a constant offset, re-run `deploy/b2w_mocap_root_calibration.py` and switch to `mocap_root_offset_path`. |
| B3 | Puck pivot → puck geometric centre | feature 12 must be the puck **centre** | 12 | `mocap_puck_offset_local` [m, puck asset frame] | §E.3 spin test and §E.4 centre test |
| B4 | Disk mount = training mount | disk frame is computed, not measured | 12, contact proxy | `disk_mount_verified: true` after checking; geometry keys must stay at training values | measure (below) + §E.4 |
| B5 | Z1 FK / arm base offset | disk frame and HIGH | 9, 12, WBC | `arm_base_offset_pos`, `z1_fk_to_policy_ee_pos` (reused from retrieval) | Model consistency verified offline (0.0001 mm). Physical accuracy (backlash, link compliance under the disk+puck load) only via §E.4. |
| B6 | Puck geometry | rest height 0.042 m and contact proxy | 12, proxy, start gate | `puck_geometry_verified: true` | calipers: radius 50 ± 2 mm, height 80 ± 2 mm |

**B3 procedure (puck centre offset)** — asset created upright (§A step 3):
* z: stand the puck upright on the calibrated floor. Its centre is at z = 0.040 m. Read the pivot
  world z `p_z` in Motive. `offset_z = 0.040 − p_z` (negative if the pivot is above the centre).
* x, y: keep the puck upright and rotate it about its own axis in place (against a fixed V-block or
  corner) through ≥ 4 angles; the pivot traces a circle around the axis. The circle centre minus
  the pivot, expressed in the asset frame, is `(offset_x, offset_y)`. Alternatively put one marker
  exactly at the top-face centre and set the Motive pivot to that marker; then `offset = [0, 0, −(0.040 + marker-centre height above the top face)]`.
* Write `mocap_puck_offset_local: [offset_x, offset_y, offset_z]`. `[0, 0, 0]` is only correct if
  the Motive pivot was placed exactly at the geometric centre.

**B4 procedure (disk mount)** — the training disk is a 200-mm rimless disk, 4 mm thick, whose
lower-face centre sits at `[0.100, 0.043, 0.000]` m in the gripperStator frame with
`rpy = [−π/2, 0, 0]` (disk normal = stator +Y, which points UP at the default arm pose), i.e.
collision centre `[0.100, 0.045, 0]`. Check on the hardware with
calipers relative to the stator mounting face, that the disk is flat (no warp > 1 mm), rigidly
attached (no wobble under a 1.5 kg puck), and that the closed gripper does not touch it. Then set
`disk_mount_verified: true`. If the real mount differs, **do not edit** `disk_center_offset_gripper`
/ `disk_*_axis_gripper` to match: the policy was trained on this geometry; fix the mount instead.

**Error budget vs. training noise (±10 mm on feature 12):** mocap ≲ 0.5 mm; puck offset (B3)
≲ 2 mm if done as above; base-asset rotation (B2) ≈ 15 mm per degree; Z1 FK/compliance unknown
until §E.4. Target total ≤ 5 mm at the disk centre (these targets are engineering guidance, not
values the policy was validated against).

---

## C. Payload and disk preparation

| Item | Training assumption | Real requirement |
|---|---|---|
| Disk | rimless, radius 0.100 m, thickness 0.004 m, rigidly merged into gripperStator at the mount above | same; no rim (the policy never saw one) |
| Puck shape | solid cylinder r = 0.050 m, h = 0.080 m, axis ∥ disk normal | same ±2 mm; flat bottom face |
| Puck mass | U(0.50, 1.50) kg | measured incl. markers, in range → `puck_mass_kg` |
| Puck–disk friction | μ_s ~ U(0.3, 1.0), μ_d ≤ μ_s | must lie in range. **For first tests use a high-friction pair (μ ≳ 0.8)**: in sim2sim the policy retains the puck 98.8 % for μ > 0.78 vs 7.8 % for μ < 0.54 (`transport_experiment/docs/ALIGNMENT_REPORT.md`). Measure μ_s as tan(θ) of the tilt angle at which the puck starts to slide on the dismounted disk. |
| Restitution | ~0 | non-bouncy materials |
| Initial placement | disk centre ±10 mm, resting, zero velocity | the start gate requires `|x_D|,|y_D| ≤ 20 mm` and `|z_D − 0.0425| ≤ 10 mm` (spawn ±10 mm + obs noise 10 mm) |
| Gripper | held at 0.0 (closed) | held at 0.0 from startup to exit; disk must clear the jaw |
| Visibility | — | puck markers visible from ≥ 3 cameras over the HIGH/LOW/LEFT/RIGHT EE poses (LOW: kp0 z = 0.35 m, 0.75 m ahead of the base) and the planned driving area. Any untracked frame → immediate damping protection. |
| Clearance | — | LEFT/RIGHT move the EE 0.30 m sideways; LOW reaches 0.35 m height in front of the robot. Keep that volume clear. |

Safety: the puck is an unsecured 0.5–1.5 kg mass up to ≈1.2 m high. Put a foam mat in the
operating area, keep people outside the arm's reach, and consider a light tether that cannot pull on
the disk.

---

## D. Hardware and network

| Setting | Value | Source |
|---|---|---|
| Unitree DDS NIC (CLI arg) | e.g. `enxa0cec819e15f` | reuse retrieval |
| Motive server / local NIC | `192.168.0.118` / `192.168.0.17`, multicast `239.255.42.99`, ports 1511/1510 | reuse retrieval YAML |
| Up axis | `auto` | reuse |
| Rigid bodies streamed | `B2` (9) + puck (new) | §A |
| Freshness | body 0.25 s (retrieval), **puck 0.10 s (= one HL period)**; age = local receive time | YAML |
| Time sync | none needed: freshness uses the receiving PC's monotonic clock; no cross-clock timestamps | — |
| Z1 SDK | `z1_sdk_lib: /home/huanyuguo/Workspace_huanyu/z1_sdk_huanyu/lib` (robot PC); z1 controller running as for retrieval | reuse |
| Models | high `exported_policies/b2wz1_hl_transport_uan/high_policy.onnx`, WBC `exported_policies/b2wz1_wbc_uan_plb/policy_wbc_uan_transport.onnx` (both sha256-pinned) | this commit |
| Python deps | same env as retrieval (`numpy`, `onnxruntime`, `pyyaml`, `unitree_sdk2py`, Z1 SDK) | reuse |
| Logs | `Log/transport/transport_<date>.csv`, one row per HL step | new |

---

## E. Preflight validation (on the robot, before pressing A to enable)

The READY-WAIT line printed once per second while the robot stands at DEFAULT shows everything
needed: `base_h`, `puck_D`, `z_err = z_D − 0.042`, `puck_age`, and the gate verdict.

1. **Config gate**: `python3 deploy/b2wz1_hl_transport_uan_mocap.py <nic> <yaml>` prints
   `Setup incomplete -- nothing was started` and exits until §F items are filled. Expected after
   setup: it proceeds to `Waiting for the first B2W lowstate`.
2. **Tracking**: Motive shows `B2` and the puck tracked; READY-WAIT shows `puck_age` < 20 ms and
   `base_h` ≈ the retrieval standing height. Cover the puck markers by hand → READY-WAIT must show
   `PUCK:UNTRACKED` (or `STALE`) and the gate must not pass.
3. **Puck centre (B3) — spin test**: with the puck on the disk centre, rotate it by hand about its
   own axis without translating it. `puck_D` x/y must change by < 2 mm. Larger → fix `mocap_puck_offset_local` x/y.
4. **Disk centre / FK / base calibration (B2, B4, B5)**: mark the physical disk centre. Place the
   puck there. Expect `puck_D = (0 ± 3, 0 ± 3, 42 ± 3) mm`, i.e. `z_err` within ±3 mm. Repeat at
   the 4 cardinal points 60 mm from the centre: `puck_D` xy must read (±60, 0)/(0, ±60) mm ± 3 mm.
   At the DEFAULT arm pose the disk frame is (to 3.4°) the robot frame: **x_D = forward,
   y_D = left, z_D = up** (in stator terms x_D = +X, y_D = −Z, z_D = +Y). A wrong sign means a
   mount or asset-axis problem. A consistent offset points to B2/B4; an error that changes with the
   arm pose points to FK/compliance (then consider the optional temporary disk rigid body).
5. **Stationary noise**: puck resting, robot standing: `puck_D` fluctuation < 1 mm over 10 s (no
   velocity is estimated, so this is only a noise check).
6. **Gate**: puck 30 mm off centre → gate reports `xy … outside ±0.020` and A is ignored. Puck back
   at centre → `gate=OK`.
7. **Models**: banner prints both policy paths with "sha256 verified"; "UAN: NOT LOADED"; Z1 gains
   `Kp=[64, 115.2, 64, 64, 64, 64]`, `Kd=[3, 4, 3, 3, 3, 3]`.
8. **Safety controls**: SELECT, Ctrl-C and your lab's power-off/E-stop procedure are known to the
   operator; a foam mat is in place; the A-press leg recovery has been rehearsed with retrieval.

After a run, check the CSV: `exec_*` within [−1, 1], `contact` = 1 throughout, `puck_rho` small,
`kp0_meas_*` following `kp0_cmd_*`.

---

## Deployment procedure

**Step 0 — validate the transport WBC alone (required, new checkpoint).** The transport WBC has
never run on hardware. With the disk mounted and **no puck**, run the validated standalone WBC test
with only `policy_path` changed in a copy of `deploy/configs/b2wz1_wbc_uan.yaml`:
```
policy_path: "exported_policies/b2wz1_wbc_uan_plb/policy_wbc_uan_transport.onnx"
```
Confirm it stands, follows stick base commands and holds its EE commands as smoothly as
`policy_wbc_uan_3.onnx` does. Also look at the disk tilt while holding the HIGH pose: the sim2sim
alignment report found this WBC tilts the disk 20–30° in MuJoCo with zero residual; whether the real
robot does the same is unknown and directly limits puck retention.

**Launch** (repo root, robot PC):
```
python3 deploy/b2wz1_hl_transport_uan_mocap.py enxa0cec819e15f deploy/configs/b2wz1_hl_transport_uan_mocap.yaml
```

**Operator sequence** (the controller only moves on explicit presses):
1. Zero torque → **START**.
2. Arm moves to DEFAULT (gripper held closed) → **A** (then release) to stand.
3. B2W squats, then stands at DEFAULT (stiff PD).
4. READY-WAIT: place the puck at the disk centre, run §E checks; when `gate=OK`, **A** = explicit enable.
5. 0.5 s runtime-gain settle, policy state initialised (HIGH measured), 0.5 s zero-residual HL
   settle (as training), then transport is active.
6. During transport: **left stick** vx/vy, **right stick** x → wz (deadband 20 %);
   **D-pad** up/down/left/right = HIGH/LOW/LEFT/RIGHT (≥ 4 s apart, else rejected and logged).
7. Stop: **SELECT** → damping protection (B2W kp=0/high kd, Z1 holds with high kd; the puck is not
   held and may fall). In protection: release A, press **A** to fold the legs; **Ctrl-C** to exit.

Automatic damping protection on: SELECT, Z1 PASSIVE, base or puck mocap untracked/stale/missing,
puck drop (proxy below), non-finite observation/action, WBC arm target outside the SDK range, any
exception. The policy never resumes after protection.

**Puck-loss detection.** Training ends the episode when the puck/disk contact force stays
≤ 0.05 N for 5 consecutive HL steps (0.5 s). There is no force sensor, so contact is
`rho_D ≤ 0.100 m and |z_D − 0.042| ≤ 0.010 m` with the same 5-step debounce. Across 8 sim2sim
episodes this proxy fired within ±1 HL step of the training detector (never missed, never extra).
It is more lenient during near-weightless phases (fast downward disk motion, where MuJoCo reports
~0 N with the puck still resting) and up to 1 step earlier when the puck tips over the rim.

**Recommended first supervised sequence**
1. Step 0 above.
2. Preflight §E with the robot standing; Ctrl-C out (one damping packet, as retrieval).
3. Enable with sticks untouched (base ref 0), stay at HIGH for 10–20 s, SELECT. Review the CSV.
4. Same, with one D-pad switch at a time (LEFT, RIGHT, then LOW — LOW is the largest motion).
5. Lower `command_scale` (e.g. `[0.2, 0.2, 0.2]`) and add slow base motion; increase toward 0.6
   only after the CSV shows `exec_*` well inside ±1 and no near-drops.
6. Only then vary puck mass toward 1.5 kg and lower friction.

---

## Test results (offline)

Run:
```
~/miniconda3/envs/env_unitree_mujoco/bin/python -B deploy/tests/test_b2wz1_hl_transport_uan_mocap.py
LD_LIBRARY_PATH=../z1_sdk/lib ~/miniconda3/envs/env_unitree_sdk2/bin/python -B deploy/tests/test_b2wz1_hl_transport_uan_mocap.py
```
(`deploy/tests/make_transport_fk_fixture.py` regenerates the MJCF FK fixture in the MuJoCo env.)
All DDS entry points are patched to fail the test if touched.

| Area | Result |
|---|---|
| Models | sha256 high `db783046…` = sim2sim `high_policy_2.onnx`; WBC `6b5aa5de…` = sim2sim `policy_wbc_uan_transport.onnx`; UAN `db53ac18…` = sim2sim UAN (provenance); 192→6 and 400→22; deterministic; outputs in [−1, 1]; wrong WBC (`47f19872…`) rejected |
| Config | all task constants equal the sim2sim YAML; shipped YAML refuses to start (5 missing setup items); `main()` exits before DDS |
| Action parity | 1000 random references/actions incl. clipping and settle: decode and 9-D keypoints **bit-identical** to sim2sim |
| Closed-loop parity | the deploy controller's `wbc_step()` driving TransportSim (bit-exact twin of the sim2sim script) for 120 HL steps (μ=1.0) and until the puck drop (μ=0.6): max deviation actor history 1.1e-5, raw/executed residual 1.1e-5, kp0 ref 1.2e-7, EE command 1.9e-6, WBC history/action 2.4e-5 (float32 rounding) |
| FK (deploy env, Z1 SDK) | Z1 SDK FK + deploy offsets vs training MJCF over 64 poses: stator / disk centre 0.0001 mm, rotation 2e-7 |
| Mocap (fake NatNet) | base-frame transform and puck offset exact; untracked / stale / not-in-stream / base-lost → invalid with reason; missing or shared rigid body rejected at init |
| Safety | SELECT, Z1 PASSIVE, puck/base mocap loss (immediate), puck drop after 5 HL steps, transient jump no drop, NaN observation, arm range, start gate, D-pad edge + min hold, fault/KeyboardInterrupt routing in `run()` |
| Timing | mean `wbc_step` 0.2–0.25 ms (fakes; excludes real Z1 FK and DDS) of a 20-ms period |

Not tested offline (hardware only): DDS/Z1 communication, real Z1 FK accuracy under load, mocap
latency/occlusion on the real puck, transport-WBC hardware behaviour, real puck–disk friction,
the full startup/protection sequence on the robot.

---

## F. Remaining manual setup

| Item | Required? | Current status | Action needed | Validation |
|---|---|---|---|---|
| Transport WBC on hardware (Step 0) | Yes | Requires on-robot validation | run `b2wz1_wbc_uan.py` with `policy_wbc_uan_transport.onnx`, disk mounted | stands/tracks like the validated WBC; note disk tilt at HIGH |
| Puck rigid body in Motive | Yes | Requires physical setup | markers on top face, asset created upright, streaming on | Motive tracks it over all EE poses |
| `mocap_puck_name` / `mocap_puck_id` | Yes | Implemented, requires configuration (null) | enter Motive name and/or ID (≠ B2/9) | init log lists both assets; §E.2 |
| `mocap_puck_offset_local` | Yes | Requires calibration (null) | §B3 | §E.3 spin test < 2 mm |
| `puck_mass_kg` | Yes | Implemented, requires configuration (null) | weigh incl. markers, must be 0.5–1.5 kg | config gate |
| `puck_geometry_verified` | Yes | Requires physical setup (false) | r = 50 ± 2 mm, h = 80 ± 2 mm | calipers |
| `disk_mount_verified` | Yes | Requires physical setup (false) | §B4 | §E.4 |
| Puck–disk friction in [0.3, 1.0] (≥ 0.8 first) | Yes | Blocked by missing information (unmeasured) | tilt test | tan θ_slip |
| B2 rigid body = base_link | Yes | Reused from existing deployment | re-check only if markers moved | §E.4 consistent offset |
| Motive ground plane / network / NIC | Yes | Reused from existing deployment | none | §E.2 |
| Disk/FK accuracy on the real arm | Yes | Requires on-robot validation | §E.4 at centre + 4 points | ≤ 3 mm |
| Foam mat / clear volume / operator briefing | Yes | Requires physical setup | §C safety | — |
| Disk rigid body | No (optional) | Not implemented in code (calibration aid) | only if §E.4 fails | — |
| `command_scale` reduced for first runs | Recommended | Implemented (0.6 = training max) | e.g. 0.2 | — |
