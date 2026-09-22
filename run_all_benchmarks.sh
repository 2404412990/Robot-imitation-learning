#!/usr/bin/env bash
# =============================================================================
# run_all_benchmarks.sh  ——  一键生成论文所有数据和图表
#
# 指标说明 (所有指标均无需外部 ground truth):
#   Table 1 (FPS):  逐步优化吞吐量 + VRAM
#   Table 2 (Stab): 稳定性消融 — Root Jumps, Accel.Var, Jerk, Foot Penetration,
#                    Joint Limit Violations
#   Table 3 (Abl):  推理优化消融
#   Self-contained:  Jerk, Foot Penetration, Joint Limit Viol., Cycle Consistency
# =============================================================================

set -euo pipefail

# 默认使用第3块显卡 (索引为2)
export GPU_ID=${GPU_ID:-3}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export REAL_PYTHON=${WHAM_PYTHON:-python}

# ── VRAM 监控 wrapper ────────────────────────────────────────────────
WRAPPER_SCRIPT="$(pwd)/.python_mem_wrapper.sh"
cat << 'EOF' > "${WRAPPER_SCRIPT}"
#!/bin/bash
MON_GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
MON_GPU_ID=$(echo $MON_GPU_ID | cut -d',' -f1)

TMP_MEM_FILE=$(mktemp)
echo 0 > "$TMP_MEM_FILE"

(
  max_mem=0
  while true; do
     mem=$(nvidia-smi -i "$MON_GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
     if [ "$mem" -gt "$max_mem" ]; then
        max_mem=$mem
        echo "$max_mem" > "$TMP_MEM_FILE"
     fi
     sleep 0.1
  done
) &
MON_PID=$!

$REAL_PYTHON "$@"
EXIT_CODE=$?

kill $MON_PID 2>/dev/null
wait $MON_PID 2>/dev/null || true

sleep 0.5
REAL_MAX_MEM=$(cat "$TMP_MEM_FILE")
rm -f "$TMP_MEM_FILE"

echo "[VRAM] peak_allocated_mb=$REAL_MAX_MEM" >&2
exit $EXIT_CODE
EOF
chmod +x "${WRAPPER_SCRIPT}"

PYTHON="${WRAPPER_SCRIPT}"
export WHAM_PYTHON="${WRAPPER_SCRIPT}"

# ── 配置 ─────────────────────────────────────────────────────────────
VIDEO_WALK=${VIDEO_WALK:-examples/Walking.mp4}
VIDEO_SITTING=${VIDEO_SITTING:-examples/Sitting.mp4}
VIDEO_RUNNING=${VIDEO_RUNNING:-examples/Running.mp4}
VIDEO_EXERCISE=${VIDEO_EXERCISE:-examples/exercise.mp4}
VIDEO_GOLF=${VIDEO_GOLF:-examples/Golf.mp4}
VIDEO_Basketball=${VIDEO_BASKETBALL:-examples/Basketball.mp4}
VIDEO_TaiChi=${VIDEO_TAICHI:-examples/TaiChi.mp4}

ROBOT=${ROBOT:-unitree_g1}
RESULTS=${RESULTS:-results}
BENCH_SEC=${BENCH_SEC:-60}

ALL_VIDEOS=("${VIDEO_WALK}" "${VIDEO_SITTING}" "${VIDEO_RUNNING}" "${VIDEO_EXERCISE}" "${VIDEO_GOLF}" "${VIDEO_Basketball}" "${VIDEO_TaiChi}")
STAB_VIDEOS=()
for v in "${ALL_VIDEOS[@]}"; do
    if [ -f "$v" ]; then
        STAB_VIDEOS+=("$v")
    else
        echo "  [SKIP] Video not found: $v"
    fi
done

if [ ${#STAB_VIDEOS[@]} -eq 0 ]; then
    echo "  [WARN] No video files found! Using defaults."
    STAB_VIDEOS=("${VIDEO_WALK}")
fi

mkdir -p "${RESULTS}"

# Resolve robot model path
case "$ROBOT" in
    unitree_g1)   ROBOT_MODEL="assets/unitree_g1/g1_mocap_29dof.xml" ;;
    unitree_h1)   ROBOT_MODEL="assets/unitree_h1/h1.xml" ;;
    unitree_h1_2) ROBOT_MODEL="assets/unitree_h1_2/h1_2.xml" ;;
    booster_t1)   ROBOT_MODEL="assets/booster_t1/T1_serial.xml" ;;
    *)            ROBOT_MODEL="${ROBOT_MODEL_PATH:-}" ;;
esac

echo "============================================================"
echo "  run_all_benchmarks.sh"
echo "  Using GPU   : ${CUDA_VISIBLE_DEVICES}"
echo "  FPS video   : ${VIDEO_WALK}"
echo "  Stab videos : ${#STAB_VIDEOS[@]} actions (${STAB_VIDEOS[*]})"
echo "  Robot       : ${ROBOT}  (model: ${ROBOT_MODEL})"
echo "  Bench secs  : ${BENCH_SEC}"
echo "============================================================"

# ── Step 1: FPS benchmarks (Table 1 & 3) ─────────────────────────────
echo ""
echo "[1/5] FPS benchmarks (Table 1 + Table 3)..."
VIDEO="${VIDEO_WALK}" ROBOT="${ROBOT}" BENCH_SEC="${BENCH_SEC}" \
    RESULTS_DIR="${RESULTS}" WHAM_PYTHON="${PYTHON}" \
    bash bench_fps.sh 2>&1 | tee "${RESULTS}/bench_fps.log"

# ── Step 2: Stabilization ablation + Self-contained metrics ──────────
echo ""
echo "[2/5] Stabilization ablation + self-contained metrics (Table 2)..."
"${PYTHON}" bench_stabilization.py \
    --videos "${STAB_VIDEOS[@]}" \
    --robot "${ROBOT}" \
    --time "${BENCH_SEC}" \
    --output_dir "${RESULTS}" \
    --robot_model "${ROBOT_MODEL}" \
    --cycle_consistency_video "${STAB_VIDEOS[0]}" \
    2>&1 | tee "${RESULTS}/bench_stab.log"

# ── Step 3: Screenshot capture + grid stitching ────────────────────────
echo ""
echo "[3/6] Screenshot grid (capture + stitch)..."
CAPTURE_VIDEO="${STAB_VIDEOS[0]}"
CAPTURE_DIR="${RESULTS}/screenshots"
GRID_OUTPUT="${RESULTS}/screenshot_grid.png"
rm -rf "${CAPTURE_DIR}"

# Run pipeline with capture enabled (uses offscreen render, headless-safe)
"${PYTHON}" handle_wham_gmr.py \
    --video "${CAPTURE_VIDEO}" \
    --robot "${ROBOT}" \
    --robot_path "${ROBOT_MODEL}" \
    --record_whamvideo \
    --capture_interval 30 \
    --capture_dir "${CAPTURE_DIR}" \
    --time 30 \
    --track \
    --no-tcp \
    --torch_device cuda \
    2>&1 | tail -20

# Build grid from captured screenshots
"${REAL_PYTHON}" capture_frames.py \
    --screenshot_dir "${CAPTURE_DIR}" \
    --output "${GRID_OUTPUT}" \
    --cols 5

# ── Step 4: Generating figures ────────────────────────────────────────
echo ""
echo "[4/6] Generating figures..."
export WHAM_PYTHON="${REAL_PYTHON}"

TRAJ_VIDEO_NAME=""
for v in "${STAB_VIDEOS[@]}"; do
    stem=$(basename "$v" | sed 's/\.[^.]*$//')
    case "$stem" in
        *[Bb]asketball*) TRAJ_VIDEO_NAME="$stem" ; break ;;
    esac
done
if [ -z "$TRAJ_VIDEO_NAME" ]; then
    for v in "${STAB_VIDEOS[@]}"; do
        stem=$(basename "$v" | sed 's/\.[^.]*$//')
        case "$stem" in
            *[Dd]anc*|*[Dd]ance*) TRAJ_VIDEO_NAME="$stem" ; break ;;
        esac
    done
fi
if [ -z "$TRAJ_VIDEO_NAME" ]; then
    TRAJ_VIDEO_NAME=$(basename "${STAB_VIDEOS[0]}" | sed 's/\.[^.]*$//')
fi
echo "  Trajectory figure video: ${TRAJ_VIDEO_NAME}"

"${REAL_PYTHON}" plot_trajectory.py \
    --npz "${RESULTS}/table2_trajectories.npz" \
    --out "${RESULTS}/fig2_trajectory.pdf" \
    --video_name "${TRAJ_VIDEO_NAME}" \
    --fps 8.0

"${REAL_PYTHON}" plot_fps_bar.py \
    --table "${RESULTS}/table1_progressive.txt" \
    --out "${RESULTS}/fig_fps_progressive.pdf"

"${REAL_PYTHON}" plot_stabilization_bar.py \
    --table "${RESULTS}/table2_stabilization.txt" \
    --out "${RESULTS}/fig_stabilization_ablation.pdf"

# ── Step 5: Summary ──────────────────────────────────────────────────
echo ""
echo "[5/6] Summary"
echo "============================================================"
echo "  Table 1 (FPS)          : ${RESULTS}/table1_progressive.txt"
echo "  Table 2 (Stabilization): ${RESULTS}/table2_stabilization.txt"
echo "  Table 3 (Ablation)     : ${RESULTS}/table3_ablation.txt"
echo "  Self-contained metrics : ${RESULTS}/table_self_contained.txt"
echo "  Fig 2 (Trajectory)     : ${RESULTS}/fig2_trajectory.pdf"
echo "  Fig FPS bar            : ${RESULTS}/fig_fps_progressive.pdf"
echo "  Fig Stab bar           : ${RESULTS}/fig_stabilization_ablation.pdf"
echo "  Screenshot grid        : ${RESULTS}/screenshot_grid.png"
echo "============================================================"
echo ""
echo "  Next: paste table values into paper.tex where FILL appears."
echo ""

# ── Step 6: Cleanup ──────────────────────────────────────────────────
rm -f "${WRAPPER_SCRIPT}"
echo "[6/6] Done."
