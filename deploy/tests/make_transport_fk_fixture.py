#!/usr/bin/env python3
"""Generate the MuJoCo reference for the Z1-FK / disk-frame parity test.

Needs `mujoco` (env_unitree_mujoco). Reads the transport training MJCF from
the read-only unitree_mujoco checkout and writes, in the BASE frame, the
gripperStator pose and the disk_collision geom pose for random arm joint
configurations inside the live Z1 SDK range:

    python3 deploy/tests/make_transport_fk_fixture.py

The test test_b2wz1_hl_transport_uan_mocap.TestZ1ForwardKinematics then runs
the deploy-side chain (Z1 SDK FK -> Z1ArmAdapter.compute_policy_ee_pose_in_base
-> DiskGeometry) in the deploy env and compares.
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MJ_ROOT = os.path.abspath(os.path.join(HERE, "../../../unitree_mujoco"))
XML = os.path.join(MJ_ROOT, "unitree_robots/b2wz1/b2wz1_disk_puck.xml")
OUT = os.path.join(HERE, "fixtures", "z1_disk_fk_reference.json")

ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
Q_LO = np.array([-2.6180, 0.0, -2.8856, -1.5184, -1.3439, -2.7925])
Q_HI = np.array([2.6180, 2.8798, 0.0, 1.5184, 1.3439, 2.7925])
DEFAULT_Q = np.array([0.0, 1.48, -1.0, -0.54, 0.0, 1.57])


def main() -> None:
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    base, stator = bid("base_link"), bid("gripperStator")
    disk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk_collision")
    qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in ARM]

    rng = np.random.default_rng(20261116)
    qs = [DEFAULT_Q] + [rng.uniform(Q_LO, Q_HI) for _ in range(63)]
    samples = []
    for q in qs:
        d.qpos[:] = 0.0
        d.qpos[0:3] = [0.3, -0.2, 0.6]
        quat = rng.normal(size=4)
        d.qpos[3:7] = quat / np.linalg.norm(quat)
        for a, v in zip(qadr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        Rb, pb = d.xmat[base].reshape(3, 3), d.xpos[base]
        to_b = lambda p: Rb.T @ (p - pb)  # noqa: E731
        samples.append({
            "q": q.tolist(),
            "stator_pos_b": to_b(d.xpos[stator]).tolist(),
            "stator_rot_b": (Rb.T @ d.xmat[stator].reshape(3, 3)).tolist(),
            "disk_pos_b": to_b(d.geom_xpos[disk]).tolist(),
            "disk_rot_b": (Rb.T @ d.geom_xmat[disk].reshape(3, 3)).tolist(),
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"xml": os.path.relpath(XML, MJ_ROOT), "mujoco": mujoco.__version__,
                   "samples": samples}, f, indent=1)
    print(f"wrote {len(samples)} samples -> {OUT}")


if __name__ == "__main__":
    main()
