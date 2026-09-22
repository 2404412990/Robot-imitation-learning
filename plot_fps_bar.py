"""
plot_fps_bar.py  ——  从 table1_progressive.txt 生成 FPS 逐步优化柱状图

用法:
  python plot_fps_bar.py \
      --table results/table1_progressive.txt \
      --out   results/fig_fps_progressive.pdf

或者直接传入数据:
  python plot_fps_bar.py \
      --fps_values 2.1 6.3 9.8 12.1 13.5 15.2 \
      --out results/fig_fps_progressive.pdf
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 标签（与论文 Table 1 行对应）
LABELS = [
    "Baseline\n(ST, FP32)",
    "+MT\n(5 threads)",
    "+FP16\n(AMP)",
    "+Frame\nskip d=2",
    "+Window\nW=8",
    "+Scale\ns=0.5\n[Full]",
]

COLORS = ["#c0c0c0", "#a8d8ea", "#7fb3d3", "#5499c7", "#2980b9", "#1a5276"]


def parse_fps_from_table(path):
    """从 table1_progressive.txt 第一个数字列解析 FPS"""
    fps_list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[RESULT]"):
                m = re.search(r'(\d+\.\d+)', line)
                if m:
                    fps_list.append(float(m.group(1)))
    return fps_list[:6]


def plot_fps(fps_values, out_path, rt_threshold=10.0):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    x = np.arange(len(fps_values))
    bars = ax.bar(x, fps_values, color=COLORS, width=0.6,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # 实时阈值线
    ax.axhline(rt_threshold, color="#e74c3c", linewidth=1.5,
               linestyle="--", zorder=4, label=f"Real-time threshold ({rt_threshold} FPS)")

    # 数值标注
    for bar, val in zip(bars, fps_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 增益标注（相对于前一行）
    for i in range(1, len(fps_values)):
        gain = fps_values[i] / fps_values[i - 1]
        ax.text(x[i], fps_values[i] / 2, f"×{gain:.1f}",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS[:len(fps_values)], fontsize=9)
    ax.set_ylabel("End-to-End Throughput (FPS)", fontsize=10)
    ax.set_title("Progressive Optimization on RTX 3090", fontsize=11, pad=10)
    ax.set_ylim(0, max(fps_values) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 总加速比标注
    total = fps_values[-1] / fps_values[0]
    ax.text(0.97, 0.95, f"Total: ×{total:.1f} speedup",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#1a5276",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf4fc", edgecolor="#2980b9"))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    print(f"Saved: {str(out).replace('.pdf', '.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=None,
                        help="Path to table1_progressive.txt")
    parser.add_argument("--fps_values", nargs="+", type=float, default=None,
                        help="Manual FPS values (6 numbers)")
    parser.add_argument("--rt_threshold", type=float, default=10.0,
                        help="Real-time threshold line (FPS)")
    parser.add_argument("--out", default="results/fig_fps_progressive.pdf")
    args = parser.parse_args()

    if args.fps_values:
        fps = args.fps_values
    elif args.table:
        fps = parse_fps_from_table(args.table)
        if len(fps) < 2:
            print(f"WARNING: only {len(fps)} FPS values found in {args.table}. "
                  "Use --fps_values to override.")
    else:
        print("ERROR: provide --table or --fps_values")
        return

    plot_fps(fps, args.out, args.rt_threshold)


if __name__ == "__main__":
    main()
