"""
bench_stabilization.py  ——  Table 2 (Temporal Stabilization Ablation)
                           + Self-contained evaluation metrics

Measures each stabilization configuration on:
  - Root Jumps/min：pelvis displacement > 0.15 m per frame
  - Accel. Variance：root position acceleration variance (m/s^2)^2
  - Jerk：dimensionless integrated squared jerk (rad/s^3)^2
  - Foot-ground penetration rate (events/min) and avg depth (mm)
  - Joint limit violation rate (violations/min)

Output:
  results/table2_stabilization.txt
  results/table2_trajectories.npz
  results/table_self_contained.txt

Usage:
  python bench_stabilization.py --video dancing.mp4 --robot unitree_g1
"""

import argparse
import copy
import os
import sys
import types
import importlib
import time
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Stabilization ablation configs
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = [
    ("No stabilization (raw WHAM)",              "raw_wham",            True,  True,  True,  1.0,  False),
    ("+ Velocity integration",                   "vel_integration",     False, True,  True,  1.0,  False),
    ("+ Displacement thresholding",              "disp_threshold",      False, False, True,  1.0,  False),
    ("+ Dual-stage quaternion clamping",         "quat_clamp",          False, False, False, 1.0,  False),
    ("+ Exp. smoothing (alpha=0.35)",            "exp_smooth",          False, False, False, 0.35, False),
    ("+ FK-grounded height correction (Full)",   "full_pipeline",       False, False, False, 0.35, True),
]

JUMP_THRESHOLD = 0.15   # m

# Robot name → MuJoCo XML mapping
ROBOT_MODEL_MAP = {
    "unitree_g1":    "assets/unitree_g1/g1_mocap_29dof.xml",
    "unitree_h1":    "assets/unitree_h1/h1.xml",
    "unitree_h1_2":  "assets/unitree_h1_2/h1_2.xml",
    "booster_t1":    "assets/booster_t1/T1_serial.xml",
}


def resolve_model_path(robot_name):
    """Resolve MuJoCo XML path from robot name."""
    if robot_name in ROBOT_MODEL_MAP:
        return str(ROOT / ROBOT_MODEL_MAP[robot_name])
    # try as direct path
    candidate = Path(robot_name)
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        f"Unknown robot '{robot_name}'. Known: {list(ROBOT_MODEL_MAP)}. "
        f"Provide a full path to the MJCF XML."
    )


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_root_jumps_per_min(root_positions, fps):
    """Returns jumps/min: frames where ||p_t - p_{t-1}|| > JUMP_THRESHOLD."""
    if len(root_positions) < 2:
        return 0.0
    deltas = np.linalg.norm(np.diff(root_positions, axis=0), axis=1)
    n_jumps = int(np.sum(deltas > JUMP_THRESHOLD))
    duration_min = len(root_positions) / fps / 60.0
    return n_jumps / max(duration_min, 1e-6)


def compute_accel_variance(root_positions, fps):
    """Returns variance of per-frame acceleration magnitude (m/s^2)^2."""
    if len(root_positions) < 3:
        return 0.0
    dt = 1.0 / fps
    vel = np.diff(root_positions, axis=0) / dt
    accel = np.diff(vel, axis=0) / dt
    accel_mag = np.linalg.norm(accel, axis=1)
    return float(np.var(accel_mag))


def compute_jerk(joint_positions, fps):
    """
    Compute dimensionless integrated squared jerk, mean across all DOFs.

    joint_positions: (N, K) where K = number of DOFs
    Returns: scalar jerk value (rad/s^3)^2
    """
    if len(joint_positions) < 4:
        return 0.0
    dt = 1.0 / fps
    jerk = np.diff(joint_positions, n=3, axis=0) / (dt ** 3)  # (N-3, K)
    # integrated squared jerk per DOF, then mean across DOFs
    integrated = 0.5 * np.sum(jerk ** 2, axis=0) * dt
    return float(np.mean(integrated))


def compute_foot_penetration(root_positions, root_quats, joint_angles,
                              model_path, fps):
    """
    Compute foot-ground penetration using MuJoCo FK.

    Returns: (events_per_min, avg_depth_mm)
    """
    try:
        import mujoco
    except ImportError:
        print("  [WARN] mujoco not installed; skipping foot penetration metric.")
        return 0.0, 0.0

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    nq = model.nq
    nv = model.nv
    nu = model.nu

    penetration_events = 0
    penetration_depths = []

    n_frames = min(len(root_positions), len(joint_angles))
    for i in range(n_frames):
        qpos = np.zeros(nq)
        # root position (Z-up in MuJoCo convention)
        qpos[0:3] = root_positions[i]
        # root orientation (w,x,y,z)
        qpos[3:7] = root_quats[i] if i < len(root_quats) else [1, 0, 0, 0]
        # joint DOFs — pad or trim to match model
        ndof = len(joint_angles[i])
        njoint = min(ndof, nq - 7)
        qpos[7:7 + njoint] = joint_angles[i][:njoint]

        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        # Check all geom positions for ground penetration (z < 0)
        min_z = float(np.min(data.geom_xpos[:, 2]))
        if min_z < -0.001:  # 1 mm threshold
            penetration_events += 1
            penetration_depths.append(abs(min_z))

    duration_min = n_frames / fps / 60.0
    events_per_min = penetration_events / max(duration_min, 1e-6)
    avg_depth_mm = float(np.mean(penetration_depths) * 1000) if penetration_depths else 0.0

    return events_per_min, avg_depth_mm


def compute_joint_limit_violations(joint_angles, model_path, fps):
    """
    Count how often IK outputs exceed the robot's physical joint range.

    Returns: violations_per_min
    """
    try:
        import mujoco
    except ImportError:
        print("  [WARN] mujoco not installed; skipping joint limit metric.")
        return 0.0

    model = mujoco.MjModel.from_xml_path(model_path)
    # jnt_range: (njnt, 2), but we only want the ones corresponding to
    # the actuators/revolute joints. Use actuator_ctrlrange for consistency
    # with what GMR outputs.
    lower = model.actuator_ctrlrange[:, 0]
    upper = model.actuator_ctrlrange[:, 1]

    joint_angles = np.asarray(joint_angles)
    ndof = joint_angles.shape[1]
    nact = min(ndof, len(lower))

    violations = 0
    n_frames = joint_angles.shape[0]
    for i in range(n_frames):
        vals = joint_angles[i, :nact]
        n_v = int(np.sum((vals < lower[:nact]) | (vals > upper[:nact])))
        violations += n_v

    duration_min = n_frames / fps / 60.0
    return violations / max(duration_min, 1e-6)


def compute_drift(root_positions, fps):
    """
    Compute cumulative XY drift magnitude from origin over the full sequence.

    Returns: (drift_m, drift_rate_m_per_min)
      drift_m: total XY displacement from origin at sequence end
      drift_rate_m_per_min: drift per minute of video
    """
    if len(root_positions) < 2:
        return 0.0, 0.0
    final_xy = root_positions[-1, :2]
    drift_m = float(np.linalg.norm(final_xy))
    duration_min = len(root_positions) / fps / 60.0
    drift_rate = drift_m / max(duration_min, 1e-6)
    return drift_m, drift_rate


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_and_collect(video, robot, smooth_alpha, height_adjust,
                             disable_vel_integration, disable_disp_thresh,
                             disable_quat_clamp, time_limit=30.0,
                             output_dir="results/stab_tmp",
                             delta_max=None):
    """
    Run handle_wham_gmr.py in a subprocess with the given ablation switches.
    Reads the generated CSV and returns all needed data.

    Returns: (root_positions, root_quats, joint_angles, fps, ik_res1, ik_res2, ik_pos_err_mm)
    """
    tmpdir = Path(output_dir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    csv_path = tmpdir / "motion.csv"
    pkl_path = tmpdir / "motion.pkl"

    env = os.environ.copy()
    env["WHAM_USE_AMP"] = "1"
    env["WHAM_DETECT_INTERVAL"] = "2"
    env["WHAM_INFER_INTERVAL"] = "2"
    env["WHAM_STREAM_SEQ_LEN"] = "8"
    env["WHAM_INPUT_SCALE"] = "0.5"
    env["BENCH_DISABLE_VEL_INTEGRATION"] = "1" if disable_vel_integration else "0"
    env["BENCH_DISABLE_DISP_THRESH"] = "1" if disable_disp_thresh else "0"
    env["BENCH_DISABLE_QUAT_CLAMP"] = "1" if disable_quat_clamp else "0"
    if delta_max is not None:
        env["BENCH_DELTA_MAX"] = str(delta_max)

    cmd = [
        sys.executable, "handle_wham_gmr.py",
        "--video", str(video),
        "--time", str(time_limit),
        "--output_dir", str(tmpdir),
        "--robot", str(robot),
        "--coord_fix", "yup_to_zup",
        "--save_path", str(pkl_path),
        "--csv_path", str(csv_path),
        "--no-camera_follow",
        "--no-tcp",
        "--torch_device", "cuda",
        "--smooth_alpha", str(smooth_alpha),
    ]
    if height_adjust:
        cmd.append("--height_adjust")
    else:
        cmd.append("--no-height_adjust")

    print(f"  [CMD] {' '.join(cmd[:6])} ... (time={time_limit}s)")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=time_limit + 120)

    # Extract FPS and IK residuals from stdout+stderr (loguru outputs to stderr)
    fps = 8.0
    ik_res1_vals = []
    ik_res2_vals = []
    ik_pos_err_vals = []
    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
    fps_vals = []
    for ln in combined_output.splitlines():
        if "FPS:" in ln:
            try:
                fps_vals.append(float(ln.split("FPS:")[1].split()[0]))
            except Exception:
                pass
        if "[IK]" in ln and "res1=" in ln:
            try:
                ik_res1_vals.append(float(ln.split("res1=")[1].split()[0]))
                ik_res2_vals.append(float(ln.split("res2=")[1].split()[0]))
                ik_pos_err_vals.append(float(ln.split("pos_err_mm=")[1].split()[0]))
            except Exception:
                pass
    if fps_vals:
        fps = float(np.median(fps_vals[-30:]))

    ik_res1 = float(np.mean(ik_res1_vals)) if ik_res1_vals else 0.0
    ik_res2 = float(np.mean(ik_res2_vals)) if ik_res2_vals else 0.0
    ik_pos_err = float(np.mean(ik_pos_err_vals)) if ik_pos_err_vals else 0.0

    if not csv_path.exists():
        print(f"  [WARN] CSV not found: {csv_path}")
        return np.zeros((1, 3)), np.zeros((1, 4)), np.zeros((1, 1)), fps, ik_res1, ik_res2, ik_pos_err

    data = np.loadtxt(str(csv_path), delimiter=",", comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]

    root_positions = data[:, :3]
    # Quaternion columns 3-6: xyzw from WHAM (scalar-last); convert to wxyz
    root_quats_xyzw = data[:, 3:7]
    root_quats = np.zeros_like(root_quats_xyzw)
    root_quats[:, 0] = root_quats_xyzw[:, 3]  # w
    root_quats[:, 1] = root_quats_xyzw[:, 0]  # x
    root_quats[:, 2] = root_quats_xyzw[:, 1]  # y
    root_quats[:, 3] = root_quats_xyzw[:, 2]  # z

    # Joint DOFs from column 7 onwards
    joint_angles = data[:, 7:] if data.shape[1] > 7 else np.zeros((len(data), 1))

    return root_positions, root_quats, joint_angles, fps, ik_res1, ik_res2, ik_pos_err


# ---------------------------------------------------------------------------
# Cycle consistency runner
# ---------------------------------------------------------------------------

def run_cycle_consistency(video, robot, time_limit, output_dir):
    """
    Run the pipeline twice on the same video and measure per-DOF variance.

    Returns: mean per-DOF standard deviation (rad)
    """
    runs = []
    for run_idx in range(2):
        out_dir = Path(output_dir) / f"cycle_run_{run_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        root_pos, root_quats, joint_angles, fps, _ik1, _ik2, _pe = run_pipeline_and_collect(
            video=video, robot=robot, smooth_alpha=0.35, height_adjust=True,
            disable_vel_integration=False, disable_disp_thresh=False,
            disable_quat_clamp=False, time_limit=time_limit,
            output_dir=str(out_dir),
        )
        runs.append(joint_angles)
        print(f"  Cycle run {run_idx + 1}/2: {len(joint_angles)} frames, FPS={fps:.1f}")

    # Align to shorter length
    min_len = min(len(r) for r in runs)
    a = runs[0][:min_len]
    b = runs[1][:min_len]
    # Per-DOF standard deviation across the two runs
    per_dof_std = np.std(np.stack([a, b], axis=0), axis=0)  # (min_len, K)
    mean_std = float(np.mean(per_dof_std))
    return mean_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+",
                        default=["examples/IMG_9732.mov"],
                        help="Path(s) to test video(s)")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--time", type=float, default=30.0,
                        help="Duration per configuration (seconds)")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--robot_model", default=None,
                        help="Path to MuJoCo XML model (auto-resolved from --robot if omitted)")
    parser.add_argument("--cycle_consistency_video", default=None,
                        help="Video to use for cycle consistency test "
                             "(default: first video in --videos)")
    parser.add_argument("--drift_only", action="store_true",
                        help="Only run full config and output drift + fidelity table (Table 6+7)")
    parser.add_argument("--sensitivity_delta", action="store_true",
                        help="Sweep DELTA_MAX values and output sensitivity table (Table 8)")
    parser.add_argument("--sensitivity_video", default=None,
                        help="Video for sensitivity sweep (default: first video in --videos)")
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.robot_model or resolve_model_path(args.robot)

    videos = args.videos if isinstance(args.videos, list) else [args.videos]
    table_path = results_dir / "table2_stabilization.txt"
    npz_path = results_dir / "table2_trajectories.npz"
    sctable_path = results_dir / "table_self_contained.txt"

    # =========================================================================
    # DRIFT-ONLY MODE (Table 6 + Table 7): Run full config on each video,
    # output drift metrics + IK fidelity (IK residual, position error).
    # =========================================================================
    if args.drift_only:
        drift_path = results_dir / "table6_drift.txt"
        fid_path = results_dir / "table7_fidelity.txt"
        drift_lines = []
        fid_lines = []
        drift_header = f"{'Sequence':<16}  {'Duration(s)':>12}  {'Drift(m)':>10}  {'Drift(m/min)':>14}"
        fid_header = f"{'Sequence':<16}  {'IK Residual 1':>14}  {'IK Residual 2':>14}  {'Pos.Err(mm)':>12}"
        print("=" * 70)
        print("DRIFT + FIDELITY MODE (Table 6 + Table 7)")
        print(f"Videos: {videos}  |  Robot: {args.robot}")
        print("=" * 70)

        for video_path in videos:
            video_name = Path(video_path).stem
            out_dir = results_dir / "runs_drift" / video_name
            print(f"\n  [{video_name}] Running full config...")
            root_pos, root_quats, joint_angles, fps, ik_res1, ik_res2, ik_pos_err = \
                run_pipeline_and_collect(
                    video=video_path, robot=args.robot, smooth_alpha=0.35,
                    height_adjust=True, disable_vel_integration=False,
                    disable_disp_thresh=False, disable_quat_clamp=False,
                    time_limit=args.time, output_dir=str(out_dir),
                )

            drift_m, drift_rate = compute_drift(root_pos, fps)
            duration_s = len(root_pos) / max(fps, 1e-6)
            drift_lines.append(
                f"{video_name:<16}  {duration_s:>12.1f}  {drift_m:>10.2f}  {drift_rate:>14.2f}")
            fid_lines.append(
                f"{video_name:<16}  {ik_res1:>14.4f}  {ik_res2:>14.4f}  {ik_pos_err:>12.2f}")

            print(f"    Drift: {drift_m:.2f}m  rate={drift_rate:.2f}m/min")
            print(f"    Fidelity: IK-res1={ik_res1:.4f}  IK-res2={ik_res2:.4f}  PosErr={ik_pos_err:.1f}mm")

        drift_output = drift_header + "\n" + "-" * len(drift_header) + "\n" + "\n".join(drift_lines) + "\n"
        fid_output = fid_header + "\n" + "-" * len(fid_header) + "\n" + "\n".join(fid_lines) + "\n"
        drift_path.write_text(drift_output)
        fid_path.write_text(fid_output)
        print(f"\nSaved: {drift_path}")
        print(f"Saved: {fid_path}")
        print("Done.")
        return

    # =========================================================================
    # SENSITIVITY MODE (Table 8): Sweep DELTA_MAX values.
    # =========================================================================
    if args.sensitivity_delta:
        sens_video = args.sensitivity_video or videos[0]
        sens_path = results_dir / "table8_sensitivity.txt"
        DELTA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30]
        print("=" * 70)
        print(f"SENSITIVITY MODE (Table 8): Sweep DELTA_MAX for {Path(sens_video).stem}")
        print(f"DELTA values: {DELTA_VALUES}")
        print("=" * 70)

        jumps_results = []
        for dv in DELTA_VALUES:
            out_dir = results_dir / "runs_sensitivity" / f"delta_{dv:.2f}"
            print(f"\n  DELTA_MAX={dv:.2f}m ...")
            root_pos, root_quats, joint_angles, fps, _ik1, _ik2, _pe = \
                run_pipeline_and_collect(
                    video=sens_video, robot=args.robot, smooth_alpha=0.35,
                    height_adjust=True, disable_vel_integration=False,
                    disable_disp_thresh=False, disable_quat_clamp=False,
                    time_limit=args.time, output_dir=str(out_dir),
                    delta_max=dv,
                )
            jumps = compute_root_jumps_per_min(root_pos, fps)
            jumps_results.append(jumps)
            print(f"    Root jumps/min = {jumps:.1f}")

        # Write sensitivity table
        header = f"{'DELTA_MAX (m)':>16}  " + "  ".join(f"{v:>12.2f}" for v in DELTA_VALUES)
        row = f"{'Root jumps/min':>16}  " + "  ".join(f"{j:>12.1f}" for j in jumps_results)
        sens_output = header + "\n" + row + "\n"
        sens_path.write_text(sens_output)
        print(f"\n{sens_output}")
        print(f"Saved: {sens_path}")
        print("Done.")
        return

    print("=" * 70)
    print("TABLE 2: Temporal Stabilization Ablation + Self-Contained Metrics")
    print(f"Videos: {videos}  |  Robot: {args.robot}  |  Model: {model_path}")
    print(f"Duration: {args.time}s each")
    print("=" * 70)

    all_metrics = {}
    all_trajectories = {}

    for video_path in videos:
        video_name = Path(video_path).stem
        print(f"\n{'='*60}")
        print(f"  VIDEO: {video_name}")
        print(f"{'='*60}")

        video_metrics = {}
        for (label, key, dis_vel, dis_disp, dis_quat, alpha, h_adj) in ABLATION_CONFIGS:
            out_dir = results_dir / "runs_stab" / video_name / key

            print(f"\n  [{video_name}] Running: {label}")
            root_pos, root_quats, joint_angles, fps, ik_res1, ik_res2, ik_pos_err = run_pipeline_and_collect(
                video=video_path,
                robot=args.robot,
                smooth_alpha=alpha,
                height_adjust=h_adj,
                disable_vel_integration=dis_vel,
                disable_disp_thresh=dis_disp,
                disable_quat_clamp=dis_quat,
                time_limit=args.time,
                output_dir=str(out_dir),
            )

            jumps = compute_root_jumps_per_min(root_pos, fps)
            accel_var = compute_accel_variance(root_pos, fps)
            jerk_val = compute_jerk(joint_angles, fps)
            pen_rate, pen_depth = compute_foot_penetration(
                root_pos, root_quats, joint_angles, model_path, fps)
            jl_viol = compute_joint_limit_violations(
                joint_angles, model_path, fps)
            drift_m, drift_rate = compute_drift(root_pos, fps)

            video_metrics[key] = (jumps, accel_var, jerk_val, pen_rate, pen_depth,
                                  jl_viol, drift_m, drift_rate, ik_res1, ik_res2, ik_pos_err)
            all_trajectories[f"{video_name}/{key}"] = root_pos

            row = (f"{label:<50}  {jumps:>10.1f}  {accel_var:>12.4f}  "
                   f"{jerk_val:>12.2f}  {pen_rate:>10.1f}  {pen_depth:>8.2f}  "
                   f"{jl_viol:>10.1f}  {drift_m:>8.2f}m  {drift_rate:>8.2f}m/min  "
                   f"IK-r1:{ik_res1:.4f}  IK-r2:{ik_res2:.4f}  PosErr:{ik_pos_err:.2f}mm")
            print(f"    → {row}")

        all_metrics[video_name] = video_metrics

    # ---- Aggregated stabilization table ----
    print(f"\n{'='*60}")
    print("  AGGREGATED RESULTS (mean ± std across videos)")
    print(f"{'='*60}")

    header = (f"{'Configuration':<50}  {'Jumps/min':>18}  {'Accel.Var.':>18}  "
              f"{'Jerk':>12}  {'Pen.(ev/min)':>12}  {'Pen.(mm)':>10}  {'Limit Viol.':>12}  "
              f"{'Drift(m)':>10}  {'Drift(m/min)':>14}")
    lines = [header]
    lines.append("-" * len(header))

    for (label, key, dis_vel, dis_disp, dis_quat, alpha, h_adj) in ABLATION_CONFIGS:
        jumps_vals = [all_metrics[v][key][0] for v in all_metrics]
        accel_vals = [all_metrics[v][key][1] for v in all_metrics]
        jerk_vals  = [all_metrics[v][key][2] for v in all_metrics]
        penr_vals  = [all_metrics[v][key][3] for v in all_metrics]
        pend_vals  = [all_metrics[v][key][4] for v in all_metrics]
        jlv_vals   = [all_metrics[v][key][5] for v in all_metrics]
        drift_vals = [all_metrics[v][key][6] for v in all_metrics]
        driftr_vals = [all_metrics[v][key][7] for v in all_metrics]

        row = (f"{label:<50}  "
               f"{np.mean(jumps_vals):>7.1f} ± {np.std(jumps_vals):<7.1f}  "
               f"{np.mean(accel_vals):>7.4f} ± {np.std(accel_vals):<7.4f}  "
               f"{np.mean(jerk_vals):>7.2f} ± {np.std(jerk_vals):<5.2f}  "
               f"{np.mean(penr_vals):>7.1f} ± {np.std(penr_vals):<5.1f}  "
               f"{np.mean(pend_vals):>7.2f} ± {np.std(pend_vals):<5.2f}  "
               f"{np.mean(jlv_vals):>7.1f} ± {np.std(jlv_vals):<5.1f}  "
               f"{np.mean(drift_vals):>7.2f} ± {np.std(drift_vals):<5.2f}  "
               f"{np.mean(driftr_vals):>7.2f} ± {np.std(driftr_vals):<5.2f}")
        print(f"  {row}")
        lines.append(row)

    table_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {table_path}")

    # ---- Self-contained metrics table (one row per video, full config) ----
    sc_header = (f"{'Sequence':<16}  {'Jerk':>12}  {'Pen.(ev/min)':>12}  "
                 f"{'Pen.(mm)':>10}  {'Limit Viol.':>12}  {'Cycle σ(rad)':>14}  "
                 f"{'Drift(m)':>10}  {'Drift(m/min)':>14}")
    sc_lines = [sc_header]
    sc_lines.append("-" * len(sc_header))

    # For the self-contained table, use the full pipeline config only
    full_key = "full_pipeline"
    for video_path in videos:
        video_name = Path(video_path).stem
        if video_name in all_metrics and full_key in all_metrics[video_name]:
            m = all_metrics[video_name][full_key]
            # m = (jumps, accel_var, jerk, pen_rate, pen_depth, jl_viol, drift_m, drift_rate)
            row = (f"{video_name:<16}  {m[2]:>12.2f}  {m[3]:>12.1f}  "
                   f"{m[4]:>10.2f}  {m[5]:>12.1f}  {'PENDING':>14}  "
                   f"{m[6]:>10.2f}  {m[7]:>14.2f}")
            sc_lines.append(row)
            print(f"  {row}")

    # Cycle consistency for each video
    updated_lines = list(sc_lines)
    for i, video_path in enumerate(videos):
        video_name = Path(video_path).stem
        print(f"\n  Running cycle consistency on: {video_name} ...")
        cycle_std = run_cycle_consistency(
            video_path, args.robot, args.time,
            str(results_dir / "cycle_consistency" / video_name))
        # Update the corresponding row (skip header and separator lines)
        row_idx = i + 2  # +2 for header and separator
        if row_idx < len(updated_lines):
            updated_lines[row_idx] = updated_lines[row_idx].replace(
                "PENDING", f"{cycle_std:.4f}")
        print(f"  Cycle consistency σ = {cycle_std:.4f} rad per DOF")
    sc_lines = updated_lines

    sctable_path.write_text("\n".join(sc_lines) + "\n")
    print(f"Saved: {sctable_path}")

    # ---- Save trajectories ----
    np.savez(str(npz_path), **{k: v for k, v in all_trajectories.items()})
    print(f"Saved: {npz_path}")
    print(f"\nDone. Processed {len(videos)} video(s).")


if __name__ == "__main__":
    main()
