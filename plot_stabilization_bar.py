"""
plot_stabilization_bar.py  ——  从 table2_stabilization.txt 生成消融柱状图
                                （Accel. Variance + Jerk 双轴）

用法:
  python plot_stabilization_bar.py \
      --table results/table2_stabilization.txt \
      --out   results/fig_stabilization_ablation.pdf

或手动传值:
  python plot_stabilization_bar.py \
      --accel  0.503 0.044 0.044 0.049 0.005 0.024 \
      --jerk   38800 38800 38800 51200 2810 2810 \
      --out    results/fig_stabilization_ablation.pdf
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHORT_LABELS = [
    "None\n(raw)",
    "+Vel\nInteg.",
    "+Disp\nThresh",
    "+Quat\nClamp",
    "+Exp\nSmooth",
    "Full\nPipeline",
]


def parse_table(path):
    """解析 table2_stabilization.txt.
    每行格式: label  jumps ± std  accel ± std  jerk ± std  ...
    取 nums[2]=Accel.Var, nums[4]=Jerk"""
    accels, jerks = [], []
    with open(path) as f:
        for line in f:
            if line.strip().startswith("=") or line.strip().startswith("-") or \
               not line.strip() or "Configuration" in line or "TABLE" in line:
                continue
            nums = re.findall(r'[\d]+\.[\d]+', line)
            if len(nums) >= 6:
                accels.append(float(nums[2]))
                jerks.append(float(nums[4]))
    if len(accels) < 2:
        raise ValueError(f"Parsed fewer than 2 rows from {path}. "
                         f"Check table format. accels={accels}, jerks={jerks}")
    print(f"[parse_table] Extracted {len(accels)} rows from {path}")
    for i, (a, j) in enumerate(zip(accels, jerks)):
        print(f"  row {i}: Accel.Var={a:.4f}  Jerk={j:.1f}")
    return accels[:6], jerks[:6]


def plot_ablation(accels, jerks, out_path):
    n = len(accels)
    x = np.arange(n)
    w = 0.36

    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax2 = ax1.twinx()

    # Accel. Variance (左轴, 蓝色)
    b1 = ax1.bar(x - w / 2, accels, width=w, color="#2980b9", alpha=0.85,
                 label="Accel. Var. (m/s$^2$)$^2$ $\downarrow$", zorder=3, edgecolor="white")
    # Jerk (右轴, 橙色)
    b2 = ax2.bar(x + w / 2, jerks, width=w, color="#e67e22", alpha=0.85,
                 label="Dim'less Jerk $\downarrow$", zorder=3, edgecolor="white")

    y1_max = max(accels) if max(accels) > 0 else 1.0
    y2_max = max(jerks) if max(jerks) > 0 else 1.0

    for bar, val in zip(b1, accels):
        offset = y1_max * 0.02
        ax1.text(bar.get_x() + bar.get_width() / 2, val + offset,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                 color="#1a5276", fontweight="bold")
    for bar, val in zip(b2, jerks):
        offset = y2_max * 0.02
        ax2.text(bar.get_x() + bar.get_width() / 2, val + offset,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                 color="#7d4c00", fontweight="bold")

    b1[-1].set_edgecolor("#1a5276")
    b1[-1].set_linewidth(2)
    b2[-1].set_edgecolor("#7d4c00")
    b2[-1].set_linewidth(2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(SHORT_LABELS[:n], fontsize=9)
    ax1.set_ylabel("Accel. Variance (m/s$^2$)$^2$ $\downarrow$", color="#2980b9", fontsize=10)
    ax2.set_ylabel("Dimensionless Jerk $\downarrow$", color="#e67e22", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    ax1.set_ylim(0, y1_max * 1.25)
    ax2.set_ylim(0, y2_max * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    ax1.set_title("Temporal Stabilization Ablation — Basketball Sequence (30 s)", fontsize=11)

    a_red = (accels[0] - accels[-1]) / accels[0] * 100 if accels[0] > 0 else 0
    j_red = (jerks[0] - jerks[-1]) / jerks[0] * 100 if jerks[0] > 0 else 0
    fig.text(0.5, -0.02,
             f"Full pipeline reduces Accel. Variance by {a_red:.0f}% "
             f"and Jerk by {j_red:.0f}% vs. no stabilization",
             ha="center", fontsize=8.5, style="italic")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    print(f"Saved: {str(out).replace('.pdf', '.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=None)
    parser.add_argument("--accel", nargs="+", type=float, default=None)
    parser.add_argument("--jerk", nargs="+", type=float, default=None)
    parser.add_argument("--out", default="results/fig_stabilization_ablation.pdf")
    args = parser.parse_args()

    if args.accel and args.jerk:
        accels, jerks = args.accel, args.jerk
    elif args.table:
        accels, jerks = parse_table(args.table)
        if len(accels) < 2:
            print("WARNING: few rows found. Use --accel / --jerk to override.")
    else:
        print("ERROR: provide --table or --accel/--jerk")
        return

    plot_ablation(accels, jerks, args.out)


if __name__ == "__main__":
    main()
