"""
capture_frames.py  ——  Generate K-column x 3-row screenshot grid for visual validation.

Usage:
  # Step 1: Run the pipeline with screenshot capture enabled
  python handle_wham_gmr.py --video examples/Walking.mp4 --robot unitree_g1 \
      --record_whamvideo --capture_interval 30 --capture_dir results/screenshots \
      --time 60

  # Step 2: Generate the grid figure from captured screenshots
  python capture_frames.py --screenshot_dir results/screenshots --output results/screenshot_grid.png --cols 5

  # Or do both at once (runs handle_wham_gmr first, then builds the grid):
  python capture_frames.py --video examples/Walking.mp4 --robot unitree_g1 \
      --interval 30 --time 60 --output results/screenshot_grid.png --cols 5
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))


def natural_sort_key(name: str):
    """Sort filenames with embedded numbers naturally."""
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def collect_frame_ids(screenshot_dir: str) -> list:
    """Return sorted list of frame IDs present in all three subdirectories."""
    orig_dir = os.path.join(screenshot_dir, "orig")
    wham_dir = os.path.join(screenshot_dir, "wham")
    mujoco_dir = os.path.join(screenshot_dir, "mujoco")

    orig_ids = set()
    for f in glob.glob(os.path.join(orig_dir, "*.png")):
        orig_ids.add(Path(f).stem)

    wham_ids = set()
    for f in glob.glob(os.path.join(wham_dir, "*.png")):
        wham_ids.add(Path(f).stem)

    mujoco_ids = set()
    for f in glob.glob(os.path.join(mujoco_dir, "*.png")):
        mujoco_ids.add(Path(f).stem)

    common = sorted(orig_ids & wham_ids & mujoco_ids, key=natural_sort_key)
    return common


def build_grid(screenshot_dir: str, output_path: str, cols: int = 5,
               row_labels=("Original Video", "WHAM Visualization", "MuJoCo Robot"),
               label_height: int = 36):
    """Generate a K-column x 3-row grid, one column per time step.

    Rows (top to bottom):
      1. Original video frame
      2. WHAM visualization (mesh overlay)
      3. MuJoCo robot (offscreen render)
    """
    frame_ids = collect_frame_ids(screenshot_dir)
    if not frame_ids:
        print(f"[ERROR] No complete frame sets found in {screenshot_dir}")
        print("  Required: orig/, wham/, mujoco/ subdirectories with matching .png files")
        return False

    # If more frames than columns, pick evenly spaced subset
    if len(frame_ids) > cols:
        indices = np.linspace(0, len(frame_ids) - 1, cols, dtype=int)
        frame_ids = [frame_ids[i] for i in indices]
    else:
        cols = len(frame_ids)

    orig_dir = os.path.join(screenshot_dir, "orig")
    wham_dir = os.path.join(screenshot_dir, "wham")
    mujoco_dir = os.path.join(screenshot_dir, "mujoco")

    # Load all frames
    rows_data = [[], [], []]
    for fid in frame_ids:
        orig = cv2.imread(os.path.join(orig_dir, f"{fid}.png"))
        wham = cv2.imread(os.path.join(wham_dir, f"{fid}.png"))
        mj = cv2.imread(os.path.join(mujoco_dir, f"{fid}.png"))

        if orig is None or wham is None or mj is None:
            print(f"  [WARN] Missing frame {fid}, skipping")
            continue

        rows_data[0].append(orig)
        rows_data[1].append(wham)
        rows_data[2].append(mj)

    n_cols = len(rows_data[0])
    if n_cols == 0:
        print("[ERROR] No valid frame triplets loaded.")
        return False

    # Resize all frames to the same dimensions (match the smallest per row)
    # Use the first WHAM frame dimensions as reference
    ref_h, ref_w = rows_data[1][0].shape[:2]
    # Cap max width to 640 per column to keep overall figure manageable
    col_w = min(ref_w, 640)
    col_h = int(ref_h * col_w / ref_w)

    def resize_to(f, w, h):
        return cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)

    grid_rows = []
    for row_idx, frames in enumerate(rows_data):
        resized = [resize_to(f, col_w, col_h) for f in frames]
        # Add label on the left side
        label_img = np.full((col_h, label_height, 3), 240, dtype=np.uint8)
        # Rotated text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = row_labels[row_idx]
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
        cv2.putText(label_img, text,
                    ((label_height - th) // 2, (col_h + tw) // 2),
                    font, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
        # Rotate label vertically
        label_img = np.rot90(label_img)
        label_img = cv2.resize(label_img, (label_height, col_h))
        # Concatenate label + frames: [label | frame_1 | frame_2 | ... | frame_k]
        row_img = np.hstack([label_img] + resized)
        grid_rows.append(row_img)

    # Concatenate rows vertically
    grid = np.vstack(grid_rows)

    # Add column headers (frame IDs)
    header_h = 28
    header = np.full((header_h, grid.shape[1], 3), 240, dtype=np.uint8)
    font_scale = 0.45
    x_offset = label_height
    for i, fid in enumerate(frame_ids):
        cv2.putText(header, f"t={fid}",
                    (x_offset + col_w // 2 - 30, header_h - 8),
                    font, font_scale, (60, 60, 60), 1, cv2.LINE_AA)
        x_offset += col_w

    # Add thin separators between rows
    sep_color = (180, 180, 180)
    sep_thickness = 2
    for i in range(1, 3):
        y = i * col_h
        cv2.line(grid, (0, y), (grid.shape[1], y), sep_color, sep_thickness)

    grid = np.vstack([header, grid])

    cv2.imwrite(output_path, grid)
    print(f"Grid saved: {output_path}  ({n_cols} columns x 3 rows, {col_w}x{col_h} per cell)")
    return True


def run_pipeline_capture(video: str, robot: str, interval: int, time_sec: float,
                         capture_dir: str, **kwargs):
    """Run handle_wham_gmr.py with screenshot capture enabled."""
    cmd = [
        sys.executable, str(ROOT / "handle_wham_gmr.py"),
        "--video", str(video),
        "--robot", str(robot),
        "--record_whamvideo",
        "--capture_interval", str(interval),
        "--capture_dir", str(capture_dir),
        "--time", str(time_sec),
        "--no-tcp",
        "--torch_device", "cuda",
    ]
    if kwargs.get("camera_follow"):
        cmd.append("--camera_follow")
    if kwargs.get("smooth_alpha"):
        cmd.extend(["--smooth_alpha", str(kwargs["smooth_alpha"])])

    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] Pipeline exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Generate K-column x 3-row screenshot grid for visual validation")
    parser.add_argument("--screenshot_dir", default="results/screenshots",
                        help="Directory with orig/, wham/, mujoco/ subdirectories")
    parser.add_argument("--output", default="results/screenshot_grid.png",
                        help="Output grid image path")
    parser.add_argument("--cols", type=int, default=5,
                        help="Number of columns in the grid (default: 5)")
    # Pipeline run options
    parser.add_argument("--video", default=None,
                        help="Run pipeline with this video before building grid")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--interval", type=int, default=30,
                        help="Capture every N frames")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Pipeline run duration (seconds)")
    parser.add_argument("--camera_follow", action="store_true",
                        help="Enable MuJoCo camera follow")
    parser.add_argument("--smooth_alpha", type=float, default=None)
    args = parser.parse_args()

    # Step 1: Run pipeline if video is specified
    if args.video:
        print(f"Step 1/2: Running pipeline with capture every {args.interval} frames...")
        rc = run_pipeline_capture(
            video=args.video,
            robot=args.robot,
            interval=args.interval,
            time_sec=args.time,
            capture_dir=args.screenshot_dir,
            camera_follow=args.camera_follow,
            smooth_alpha=args.smooth_alpha,
        )
        if rc != 0:
            print(f"[WARN] Pipeline exited with code {rc}; attempting grid generation anyway.")

    # Step 2: Build grid
    print(f"Step 2/2: Building {args.cols}-column grid from {args.screenshot_dir}...")
    ok = build_grid(args.screenshot_dir, args.output, cols=args.cols)
    if ok:
        print("Done.")
    else:
        print("Failed to build grid. Check that screenshots were captured.")
        sys.exit(1)


if __name__ == "__main__":
    main()
