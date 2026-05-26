"""
plot_trajectory.py  ——  Fig 2: 骨盆 XY 轨迹对比图（有/无速度积分）

用法:
  # 方法 A: 直接从两个 CSV 文件读取（推荐）
  python plot_trajectory.py \
      --csv_no_stab  results/runs_stab/No_stabilization_raw_WHAM/motion.csv \
      --csv_full     results/runs_stab/FK-grounded_height_correction_Full/motion.csv \
      --out          results/fig2_trajectory.pdf

  # 方法 B: 从 bench_stabilization.py 生成的 npz 读取
  python plot_trajectory.py \
      --npz results/table2_trajectories.npz \
      --out results/fig2_trajectory.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection


def load_csv_positions(path):
    """从 CSV 读取根节点 XY 坐标（前两列）"""
    data = np.loadtxt(str(path), delimiter=",", comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, :2]   # (N, 2) — X, Y


def load_npz_positions(npz_path, key):
    d = np.load(str(npz_path))
    arr = d[key]
    return arr[:, :2]


def colored_line(ax, x, y, cmap="plasma", lw=1.5, alpha=0.9):
    """用时间颜色渐变绘制折线段"""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    t = np.linspace(0, 1, len(segs))
    lc = LineCollection(segs, array=t, cmap=cmap, linewidth=lw, alpha=alpha)
    ax.add_collection(lc)
    return lc


def plot_comparison(pos_no_stab, pos_full, out_path, fps=8.0, video_name="Basketball"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=150)
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.96, top=0.88, bottom=0.12)

    datasets = [
        (pos_no_stab, axes[0], "viridis",
         r"(a) Raw WHAM output (no stabilization)",
         "Sawtooth resets to origin every $\\sim W$ frames"),
        (pos_full,    axes[1], "plasma",
         r"(b) Full pipeline (velocity-integrated continuity)",
         "Smooth, continuous world-frame trajectory"),
    ]

    # Compute shared xy range so both subplots render at equal size with aspect="equal"
    x_all = np.concatenate([pos_no_stab[:, 0], pos_full[:, 0]])
    y_all = np.concatenate([pos_no_stab[:, 1], pos_full[:, 1]])
    shared_xlim = (x_all.min() - 0.3, x_all.max() + 0.3)
    shared_ylim = (y_all.min() - 0.3, y_all.max() + 0.3)

    for (pos, ax, cmap, title, subtitle) in datasets:
        x, y = pos[:, 0], pos[:, 1]
        lc = colored_line(ax, x, y, cmap=cmap, lw=1.8)
        # 起点和终点标记
        ax.plot(x[0], y[0], "go", ms=6, label="Start", zorder=5)
        ax.plot(x[-1], y[-1], "rs", ms=6, label="End", zorder=5)

        ax.set_xlim(*shared_xlim)
        ax.set_ylim(*shared_ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("World X (m)", fontsize=10)
        ax.set_ylabel("World Y (m)", fontsize=10)
        ax.set_title(title + "\n" +  subtitle , fontsize=9, pad=6)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

        # 颜色条（时间轴）
        duration = len(pos) / fps
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=duration))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Time (s)", fontsize=8)

    fig.suptitle(f"Pelvis XY Trajectory — 30 s {video_name} Sequence", fontsize=11, y=0.97)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    print(f"Saved: {str(out).replace('.pdf', '.png')}")


def main():
    parser = argparse.ArgumentParser()
    # 方法 A
    parser.add_argument("--csv_no_stab", default=None,
                        help="CSV from No-stabilization run")
    parser.add_argument("--csv_full", default=None,
                        help="CSV from Full-pipeline run")
    # 方法 B
    parser.add_argument("--npz", default=None,
                        help="NPZ from bench_stabilization.py")
    parser.add_argument("--key_no_stab", default="raw_wham",
                        help="Key in NPZ for no-stabilization trajectory")
    parser.add_argument("--key_full", default="full_pipeline",
                        help="Key in NPZ for full-pipeline trajectory")
    parser.add_argument("--video_name", default=None,
                        help="Video stem prefix for keys (e.g., 'IMG_9732/raw_wham'). "
                             "If set, prepended to --key_no_stab and --key_full.")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--out", default="results/fig2_trajectory.pdf")
    args = parser.parse_args()

    if args.csv_no_stab and args.csv_full:
        pos_no = load_csv_positions(args.csv_no_stab)
        pos_full = load_csv_positions(args.csv_full)
    elif args.npz:
        key_no = f"{args.video_name}/{args.key_no_stab}" if args.video_name else args.key_no_stab
        key_full = f"{args.video_name}/{args.key_full}" if args.video_name else args.key_full
        pos_no = load_npz_positions(args.npz, key_no)
        pos_full = load_npz_positions(args.npz, key_full)
    else:
        print("ERROR: provide either --csv_no_stab/--csv_full or --npz")
        sys.exit(1)

    # 对齐长度
    n = min(len(pos_no), len(pos_full))
    video_label = args.video_name or "Basketball"
    plot_comparison(pos_no[:n], pos_full[:n], args.out, args.fps, video_label)


if __name__ == "__main__":
    main()
