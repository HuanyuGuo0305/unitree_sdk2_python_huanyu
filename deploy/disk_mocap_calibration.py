#!/usr/bin/env python3
"""
Set up the Motive DISK rigid body for the transport sim2real.

The transport actor observes the puck centre in the TRAINING disk frame D:

    origin  disk geometric centre at MID-THICKNESS (thickness/2 below the top face)
    +X_D    in the disk plane, along the gripper, towards the fingertips
    +Z_D    disk normal, out of the top face (towards the puck)
    +Y_D    Z_D x X_D  (looking down on the top face with +X pointing away
            from you, +Y points to your LEFT)

With disk_frame_source: "mocap" the Motive disk asset's pivot and axes must BE
frame D. This tool measures frame D in the asset's own (local) coordinates
with a single loose PROBE marker, and prints the pivot translation/rotation to
apply in Motive. It needs only Motive -- not the robot, not Z1 FK.

Modes
-----
probe (default)
    You touch the probe marker (a single marker that belongs to no asset) to
    three points on the TOP FACE, pressing Enter for each:
        1. the disk centre (mark it on the print)
        2. a point on the +X_D line, ~60-90 mm from the centre
        3. a point on the +Y_D side,  ~60-90 mm from the centre
    Each point is averaged over many frames and stored in ASSET-LOCAL
    coordinates, so the arm/disk may move between points (only the probe must
    sit still on the disk while it is being sampled). Output:
        - frame D expressed in the asset frame (= the pivot change for Motive)
        - which asset axis each D axis currently is (catches 90-deg swaps,
          e.g. an asset created level in a Y-up Motive has +Y_asset = up)
        - PASS when |origin| < 1 mm and the angle < 0.5 deg (asset already = D)
    Results are saved to Log/disk_calibration/.

live
    Prints the probe marker in the asset frame, treated as D (plus the YAML
    mocap_disk_offset_local), i.e. what the transport script will compute for
    a point. Touch the probe to the centre: expect (0, 0, probe_height +
    thickness/2); on the +X line: (r, 0, same z); on the +Y side: (0, r, ...).

Asset-local coordinates are exactly Motive's: utils/natnet_client.py converts
Y-up to Z-up by PRE-multiplying the world rotation, which leaves every asset's
local frame untouched.

Run from the repository root (Motive streaming: Rigid Bodies AND
Unlabeled Markers on):

    python3 deploy/disk_mocap_calibration.py --probe-height-mm 7.5
    python3 deploy/disk_mocap_calibration.py --probe-height-mm 7.5 --disk-name disk
    python3 deploy/disk_mocap_calibration.py --mode live --probe-height-mm 7.5

--probe-height-mm is the height of the probe MARKER CENTRE above the surface it
sits on (marker radius + base); measure it with calipers. Its error goes 1:1
into the disk origin height.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_DEPLOY_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.mocap_frames import q_to_mat  # noqa: E402

DEFAULT_CONFIG = "deploy/configs/b2wz1_hl_transport_uan_mocap.yaml"

PROBES = (
    ("centre", "the disk CENTRE (top face)"),
    ("+X", "the +X line: along the gripper towards the fingertips, ~60-90 mm from the centre"),
    ("+Y", "the +Y side: to the LEFT of +X when looking down on the top face, ~60-90 mm out"),
)

PASS_POS_M = 0.001
PASS_ANGLE_DEG = 0.5


# ======================================================================
# Pure geometry (unit-tested offline)
# ======================================================================


def fit_disk_frame(
    p_centre: np.ndarray,
    p_x: np.ndarray,
    p_y: np.ndarray,
    probe_height: float,
    disk_thickness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Frame D from three probe-marker centres (all in the same frame).

    Returns (origin, R) with R = [x_D y_D z_D] as columns.
        z_D    normal of the plane through the three probes, oriented so that
               (+X probe) x (+Y probe) is positive -> out of the top face
        x_D    centre -> +X probe, projected into the plane
        origin centre probe moved down by probe_height + thickness/2
    """
    c = np.asarray(p_centre, dtype=np.float64).reshape(3)
    a = np.asarray(p_x, dtype=np.float64).reshape(3) - c
    b = np.asarray(p_y, dtype=np.float64).reshape(3) - c
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 0.03:
        raise ValueError("The +X / +Y probes must be at least 30 mm from the centre probe.")
    z = np.cross(a, b)
    nz = float(np.linalg.norm(z))
    if nz < 1e-9 or nz / (np.linalg.norm(a) * np.linalg.norm(b)) < math.sin(math.radians(30.0)):
        raise ValueError("The +X and +Y probes are (nearly) collinear with the centre.")
    z /= nz
    x = a - float(np.dot(a, z)) * z
    x /= float(np.linalg.norm(x))
    y = np.cross(z, x)
    origin = c - (float(probe_height) + 0.5 * float(disk_thickness)) * z
    return origin, np.stack([x, y, z], axis=1)


def rpy_deg(R: np.ndarray) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll) -> (roll, pitch, yaw) in degrees."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return np.degrees([roll, pitch, yaw])


def axis_angle_deg(R: np.ndarray) -> Tuple[np.ndarray, float]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    angle = math.acos(float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(angle - math.pi) < 1e-6:
        w, V = np.linalg.eigh(0.5 * (R + R.T))
        return V[:, int(np.argmax(w))], 180.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / float(np.linalg.norm(axis)), math.degrees(angle)


def nearest_axis_label(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    i = int(np.argmax(np.abs(v)))
    dev = math.degrees(math.acos(float(np.clip(abs(v[i]), -1.0, 1.0))))
    return f"{'+' if v[i] >= 0 else '-'}{'XYZ'[i]}_asset (off by {dev:.1f} deg)"


def describe_correction(origin: np.ndarray, R: np.ndarray) -> Dict[str, Any]:
    axis, angle = axis_angle_deg(R)
    return {
        "origin_asset_mm": (np.asarray(origin) * 1e3).tolist(),
        "R_asset_D": np.asarray(R).tolist(),
        "rpy_deg_zyx": rpy_deg(R).tolist(),
        "axis_angle": {"axis": axis.tolist(), "angle_deg": angle},
        "axis_map": {name: nearest_axis_label(R[:, i]) for i, name in enumerate(("X_D", "Y_D", "Z_D"))},
        "pass": bool(np.linalg.norm(origin) < PASS_POS_M and angle < PASS_ANGLE_DEG),
    }


# ======================================================================
# Mocap
# ======================================================================


def load_mocap_settings(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def make_client(cfg: Dict[str, Any]):
    from utils.natnet_client import NatNetClient  # noqa: PLC0415

    return NatNetClient(
        server_ip=cfg.get("mocap_server_ip") or None,
        local_ip=str(cfg.get("mocap_local_ip", "")),
        multicast_group=str(cfg.get("mocap_multicast_group", "239.255.42.99")),
        data_port=int(cfg.get("mocap_data_port", 1511)),
        command_port=int(cfg.get("mocap_command_port", 1510)),
        join_multicast=bool(cfg.get("mocap_join_multicast", True)),
        up_axis=str(cfg.get("mocap_up_axis", "auto")),
        verbose=False,
    )


def resolve_disk_id(client, name: Optional[str], rb_id: Optional[int]) -> int:
    names = client.rigid_body_names()
    streamed = client.rigid_body_ids()
    listing = ", ".join(f"{i}:{names.get(i, '?')!r}" for i in streamed) or "(none)"
    print(f"[DISK-CAL] streaming rigid bodies: {listing}")
    if name:
        for i, n in names.items():
            if n == name:
                return int(i)
        if rb_id is None:
            raise RuntimeError(f"Rigid body {name!r} not found in Motive's model definitions.")
        print(f"[DISK-CAL][WARN] name {name!r} not resolvable; using id {rb_id}")
    if rb_id is None:
        raise RuntimeError(
            "No disk asset given: pass --disk-name/--disk-id or set mocap_disk_name/id in the YAML."
        )
    if int(rb_id) not in streamed:
        raise RuntimeError(f"Rigid body id {rb_id} is not being streamed (streamed: {streamed}).")
    return int(rb_id)


def probe_in_asset(frame, disk_id: int, search_radius: float):
    """(probe in asset-local coords, status) for one frame."""
    rb = frame.rigid_bodies.get(disk_id) if frame is not None else None
    if rb is None:
        return None, "disk not in frame"
    if not rb.tracking_valid:
        return None, "disk UNTRACKED"
    pts = np.asarray(frame.unlabeled_markers, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return None, "no unlabeled markers (Motive: stream Unlabeled Markers?)"
    dist = np.linalg.norm(pts - rb.pos, axis=1)
    near = pts[dist <= search_radius]
    if near.shape[0] != 1:
        return None, f"{near.shape[0]} unlabeled markers within {search_radius * 1e3:.0f} mm (need exactly 1)"
    R = q_to_mat(rb.quat_wxyz)
    return R.T @ (near[0] - rb.pos), "OK"


def sample_probe(client, disk_id: int, n: int, search_radius: float, timeout_s: float) -> np.ndarray:
    samples, last_frame, last_status = [], None, ""
    t_end = time.monotonic() + timeout_s
    while len(samples) < n:
        if time.monotonic() > t_end:
            raise RuntimeError(f"timed out after {len(samples)}/{n} samples: {last_status}")
        frame = client.latest_frame()
        if frame is None or frame.frame_number == last_frame:
            time.sleep(0.002)
            continue
        last_frame = frame.frame_number
        p, last_status = probe_in_asset(frame, disk_id, search_radius)
        if p is not None:
            samples.append(p)
    arr = np.asarray(samples)
    std_mm = float(np.linalg.norm(arr.std(axis=0)) * 1e3)
    if std_mm > 1.0:
        raise RuntimeError(f"probe moved while sampling (std {std_mm:.2f} mm > 1 mm); hold it still")
    print(f"           {n} frames, std {std_mm:.2f} mm, asset-local = "
          f"({arr.mean(0)[0] * 1e3:+.1f}, {arr.mean(0)[1] * 1e3:+.1f}, {arr.mean(0)[2] * 1e3:+.1f}) mm")
    return arr.mean(axis=0)


def print_correction(corr: Dict[str, Any]) -> None:
    o = corr["origin_asset_mm"]
    rpy = corr["rpy_deg_zyx"]
    aa = corr["axis_angle"]
    print("\n" + "=" * 88)
    print("Frame D measured in the CURRENT disk asset frame (Motive asset-local coordinates)")
    print("=" * 88)
    print(f"D origin (where the pivot must go) : ({o[0]:+.1f}, {o[1]:+.1f}, {o[2]:+.1f}) mm")
    print("D axes, currently:")
    for k, v in corr["axis_map"].items():
        print(f"   {k} = {v}")
    print(f"Rotation asset -> D                : rpy = ({rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}) deg "
          "[R = Rz(yaw) Ry(pitch) Rx(roll)]")
    print(f"                                     axis-angle = {aa['angle_deg']:.2f} deg about "
          f"({aa['axis'][0]:+.3f}, {aa['axis'][1]:+.3f}, {aa['axis'][2]:+.3f})")
    if corr["pass"]:
        print(f"\nPASS: the asset already IS frame D (|origin| < {PASS_POS_M * 1e3:.0f} mm, "
              f"angle < {PASS_ANGLE_DEG} deg).")
    else:
        print("\nNOT ALIGNED. In Motive, edit the disk asset's pivot: translate it by the D origin "
              "above,\nthen rotate it so that the asset axes become the D axes. Re-run this tool "
              "afterwards:\nthe residual must shrink to PASS (if it grows, Motive applied the "
              "rotation with the\nopposite sign/order -- undo and enter it the other way).")
    print("=" * 88)


def run_probe(args, cfg) -> None:
    thickness = float(cfg.get("disk_thickness", 0.004))
    client = make_client(cfg)
    client.start()
    try:
        if not client.wait_for_first_frame(timeout_s=10.0):
            raise RuntimeError("No NatNet frames received (Motive streaming / network).")
        disk_id = resolve_disk_id(client, args.disk_name, args.disk_id)
        print(f"[DISK-CAL] disk asset id {disk_id}; probe height {args.probe_height_mm:.2f} mm; "
              f"disk thickness {thickness * 1e3:.1f} mm")
        print("[DISK-CAL] Only the probe marker may be loose within "
              f"{args.search_radius * 1e3:.0f} mm of the disk pivot.\n")
        pts = {}
        for key, text in PROBES:
            input(f"  Place the probe on {text}, hold it still, press Enter ")
            pts[key] = sample_probe(client, disk_id, args.samples, args.search_radius, timeout_s=15.0)
        for key in ("+X", "+Y"):
            r = np.linalg.norm(pts[key] - pts["centre"]) * 1e3
            print(f"  {key} probe is {r:.1f} mm from the centre probe")
        origin, R = fit_disk_frame(pts["centre"], pts["+X"], pts["+Y"],
                                   args.probe_height_mm * 1e-3, thickness)
        corr = describe_correction(origin, R)
        print_correction(corr)

        out_dir = os.path.join(PROJECT_ROOT, "Log", "disk_calibration")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, time.strftime("disk_cal_%Y%m%d_%H%M%S.json"))
        with open(out, "w") as f:
            json.dump({
                "disk_id": disk_id, "probe_height_mm": args.probe_height_mm,
                "disk_thickness_m": thickness,
                "probes_asset_m": {k: v.tolist() for k, v in pts.items()},
                **corr,
            }, f, indent=2)
        print(f"[DISK-CAL] saved {out}")
    finally:
        client.stop()


def run_live(args, cfg) -> None:
    offset = np.asarray(cfg.get("mocap_disk_offset_local") or [0.0, 0.0, 0.0], dtype=np.float64)
    thickness = float(cfg.get("disk_thickness", 0.004))
    z_expect = args.probe_height_mm + 0.5 * thickness * 1e3
    client = make_client(cfg)
    client.start()
    try:
        if not client.wait_for_first_frame(timeout_s=10.0):
            raise RuntimeError("No NatNet frames received (Motive streaming / network).")
        disk_id = resolve_disk_id(client, args.disk_name, args.disk_id)
        print(f"[DISK-CAL] live probe in frame D (asset + offset {np.round(offset * 1e3, 1).tolist()} mm). "
              f"Centre should read (0, 0, {z_expect:.1f}) mm. Ctrl-C to stop.")
        while True:
            p, status = probe_in_asset(client.latest_frame(), disk_id, args.search_radius)
            if p is None:
                print(f"\r  {status:<70s}", end="", flush=True)
            else:
                d = (p - offset) * 1e3
                print(f"\r  probe_D = ({d[0]:+7.1f}, {d[1]:+7.1f}, {d[2]:+7.1f}) mm  "
                      f"rho = {math.hypot(d[0], d[1]):6.1f} mm  z-{z_expect:.1f} = {d[2] - z_expect:+5.1f} mm   ",
                      end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        client.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the Motive disk asset against training frame D.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="transport YAML (mocap network + disk asset)")
    parser.add_argument("--mode", choices=("probe", "live"), default="probe")
    parser.add_argument("--probe-height-mm", type=float, required=True,
                        help="probe marker CENTRE height above the surface it sits on [mm]")
    parser.add_argument("--disk-name", default=None, help="override mocap_disk_name")
    parser.add_argument("--disk-id", type=int, default=None, help="override mocap_disk_id")
    parser.add_argument("--samples", type=int, default=120, help="frames averaged per probe point")
    parser.add_argument("--search-radius", type=float, default=0.25,
                        help="probe must be the only unlabeled marker within this distance of the pivot [m]")
    args = parser.parse_args()

    cfg = load_mocap_settings(args.config)
    if args.disk_name is None and args.disk_id is None:
        args.disk_name = cfg.get("mocap_disk_name")
        args.disk_id = cfg.get("mocap_disk_id")
    (run_live if args.mode == "live" else run_probe)(args, cfg)


if __name__ == "__main__":
    main()
