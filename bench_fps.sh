#!/usr/bin/env bash
# =============================================================================
# bench_fps.sh  ——  Table 1 (Progressive Optimization) + Table 3 (Ablation)
#
# 用法（Ubuntu 服务器 / 本地）:
#   conda activate wham_gmr
#   VIDEO=your_video.mp4 bash bench_fps.sh
#
# 服务器无头模式（推荐）: 不需要 DISPLAY，不开任何渲染窗口
# =============================================================================

set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Python 解释器
PYTHON=${WHAM_PYTHON:-$(which python)}

# IK iterations per stage — match run.sh default for real-time throughput
export GMR_MAX_ITER=${GMR_MAX_ITER:-5}

VIDEO=${VIDEO:-examples/IMG_9732.mov}
ROBOT=${ROBOT:-unitree_g1}
BENCH_SEC=${BENCH_SEC:-60}
RESULTS_DIR=${RESULTS_DIR:-results}

mkdir -p "${RESULTS_DIR}/runs"

TABLE1="${RESULTS_DIR}/table1_progressive.txt"
TABLE3="${RESULTS_DIR}/table3_ablation.txt"

echo "=== bench_fps.sh ==="
echo "Python  : ${PYTHON}"
echo "Video   : ${VIDEO}"
echo "Robot   : ${ROBOT}"
echo "BenchSec: ${BENCH_SEC}"
echo "Results : ${RESULTS_DIR}"
echo ""

# ---------------------------------------------------------------------------
# run_and_measure TAG [extra args...]
#   → 输出一行: "FPS WHAM_LAT_MS VRAM_GB"
# ---------------------------------------------------------------------------
run_and_measure() {
    local tag="$1"; shift
    local log="${RESULTS_DIR}/runs/${tag}.log"
    local out_dir="${RESULTS_DIR}/runs/${tag}"
    mkdir -p "${out_dir}"

    echo "  [RUN] ${tag}  (env: AMP=${WHAM_USE_AMP:-0} DET=${WHAM_DETECT_INTERVAL:-1} INF=${WHAM_INFER_INTERVAL:-1} W=${WHAM_STREAM_SEQ_LEN:-16} S=${WHAM_INPUT_SCALE:-1.0})"

    # NOTE: no --record_gmrvideo → MuJoCo viewer not launched (avoids headless hangs)
    # --torch_device cuda moves FK/kinematics to GPU
    # Use tee so terminal sees progress in real-time while log is still saved
    "${PYTHON}" handle_wham_gmr.py \
        --video "${VIDEO}" \
        --time "${BENCH_SEC}" \
        --output_dir "${out_dir}" \
        --robot "${ROBOT}" \
        --coord_fix yup_to_zup \
        --save_path "${out_dir}/motion.pkl" \
        --csv_path "${out_dir}/motion.csv" \
        --no-camera_follow \
        --no-tcp \
        --torch_device cuda \
        "$@" \
        2>&1 | tee "${log}" || true

    # VRAM from process-level torch.cuda.max_memory_allocated (logged at exit)
    local vram_mb vram_gb
    vram_mb=$(grep -oP '(?<=peak_allocated_mb=)[0-9]+' "${log}" 2>/dev/null | tail -1 || echo "0")
    vram_mb=${vram_mb:-0}
    if [[ "${vram_mb}" != "0" ]]; then
        vram_gb=$(awk "BEGIN{printf \"%.1f\", ${vram_mb}/1024}" 2>/dev/null || echo "N/A")
    else
        vram_gb="N/A"
    fi

    # 从日志中提取 FPS：handle_wham_gmr 每帧打印 "FPS: 12.3 | Latency: ..."
    local fps
    fps=$(grep -oP '(?<=FPS:\s)[0-9]+\.[0-9]+' "${log}" 2>/dev/null | tail -20 | \
          awk '{s+=$1; n++} END{if(n>0) printf "%.1f", s/n; else print "N/A"}')

    # WHAM 推理延迟（ms）—— 单独计时行，格式: [WHAM] infer_ms=45.2
    local wham_lat
    wham_lat=$(grep -oP '(?<=infer_ms=)[0-9]+\.[0-9]+' "${log}" 2>/dev/null | tail -20 | \
               awk '{s+=$1; n++} END{if(n>0) printf "%.0f", s/n; else print "N/A"}')

    printf "  FPS=%-6s  WHAM_LAT=%-6s ms  VRAM=%s GB\n" \
           "${fps}" "${wham_lat}" "${vram_gb}" >&2

    printf "%s\t%s\t%s" "[RESULT]${fps}" "${wham_lat}" "${vram_gb}"
}

# =============================================================================
# TABLE 1: 逐步叠加优化
# =============================================================================
{
echo "==========================================================================="
echo "TABLE 1: Progressive Optimization — RTX 3090 / Video: ${VIDEO}"
echo "==========================================================================="
printf "%-52s\t%6s\t%13s\t%8s\n" "Configuration" "FPS" "WHAM Lat(ms)" "VRAM(GB)"
echo "---------------------------------------------------------------------------"
} | tee "${TABLE1}"

row1() {
    local label="$1"; shift
    printf "%-52s\t" "${label}" | tee -a "${TABLE1}"
    run_and_measure "$@" | tee -a "${TABLE1}"
    echo "" | tee -a "${TABLE1}"
}

# 行1: Baseline — 单线程, FP32, W=16, d=1, s=1.0, 无高度矫正
WHAM_USE_AMP=0 WHAM_DETECT_INTERVAL=1 WHAM_INFER_INTERVAL=1 \
WHAM_STREAM_SEQ_LEN=16 WHAM_INPUT_SCALE=1.0 \
row1 "Baseline (single-thread, FP32, W=16, d=1, s=1.0)" \
     "t1_01_baseline" \
     --smooth_alpha 1.0 --no-height_adjust

# 行2: + 多线程 (handle_wham_gmr 已经是多线程，行1行2可用同参数跑两次对比)
# 实际上行1=行2，benchmark 里这一行从行1抄数据；或者对比 demo.py(单线程) vs handle_wham_gmr.py
WHAM_USE_AMP=0 WHAM_DETECT_INTERVAL=1 WHAM_INFER_INTERVAL=1 \
WHAM_STREAM_SEQ_LEN=16 WHAM_INPUT_SCALE=1.0 \
row1 "  + Multi-threading (5 parallel threads)" \
     "t1_02_mt" \
     --smooth_alpha 1.0 --no-height_adjust

# 行3: + FP16
WHAM_USE_AMP=1 WHAM_DETECT_INTERVAL=1 WHAM_INFER_INTERVAL=1 \
WHAM_STREAM_SEQ_LEN=16 WHAM_INPUT_SCALE=1.0 \
row1 "  + FP16 mixed precision (AMP)" \
     "t1_03_fp16" \
     --smooth_alpha 1.0 --no-height_adjust

# 行4: + 跳帧 d=2
WHAM_USE_AMP=1 WHAM_DETECT_INTERVAL=2 WHAM_INFER_INTERVAL=2 \
WHAM_STREAM_SEQ_LEN=16 WHAM_INPUT_SCALE=1.0 \
row1 "  + Frame skipping (d_det=2, d_inf=2)" \
     "t1_04_skip" \
     --smooth_alpha 1.0 --no-height_adjust

# 行5: + 短窗口 W=8
WHAM_USE_AMP=1 WHAM_DETECT_INTERVAL=2 WHAM_INFER_INTERVAL=2 \
WHAM_STREAM_SEQ_LEN=8 WHAM_INPUT_SCALE=1.0 \
row1 "  + Reduced temporal window (W=8)" \
     "t1_05_w8" \
     --smooth_alpha 1.0 --no-height_adjust

# 行6: + 缩放 s=0.5  [Full Optimized]
WHAM_USE_AMP=1 WHAM_DETECT_INTERVAL=2 WHAM_INFER_INTERVAL=2 \
WHAM_STREAM_SEQ_LEN=8 WHAM_INPUT_SCALE=0.5 \
row1 "  + Input scaling (s=0.5)  [FULL OPTIMIZED]" \
     "t1_06_full" \
     --smooth_alpha 0.35 --height_adjust

echo "===========================================================================" | tee -a "${TABLE1}"
echo "Saved: ${TABLE1}"

# 取全优化 FPS 用于 Table 3 计算 ΔFPS
FULL_LINE=$(grep "FULL OPTIMIZED" "${TABLE1}" | tail -1 || echo "")
FULL_FPS=$(echo "${FULL_LINE}" | grep -oP '[0-9]+\.[0-9]+' | head -1 || echo "0")
echo "Full-optimized FPS = ${FULL_FPS}"

# =============================================================================
# TABLE 3: 消融——逐一去掉一个优化（其他保持全优化状态）
# =============================================================================
{
echo ""
echo "==========================================================================="
echo "TABLE 3: Inference Optimization Ablation — Full config baseline FPS=${FULL_FPS}"
echo "==========================================================================="
printf "%-45s\t%6s\t%8s\t%8s\n" "Configuration" "FPS" "ΔFPS" "VRAM(GB)"
echo "---------------------------------------------------------------------------"
} | tee "${TABLE3}"

row3() {
    local label="$1"; shift
    local res fps delta vram
    res=$(run_and_measure "$@")
    fps=$(echo "${res}" | cut -f1)
    vram=$(echo "${res}" | cut -f3)
    delta=$(awk "BEGIN{printf \"%.1f\", ${FULL_FPS}-(${fps}+0)}" 2>/dev/null || echo "N/A")
    printf "%-45s\t%6s\t%8s\t%8s\n" "${label}" "${fps}" "-${delta}" "${vram}" | tee -a "${TABLE3}"
}

# Full (参考行)
WHAM_USE_AMP=1 WHAM_DETECT_INTERVAL=2 WHAM_INFER_INTERVAL=2 \
WHAM_STREAM_SEQ_LEN=8 WHAM_INPUT_SCALE=0.5 \
row3 "Full optimized" "t3_00_full" --smooth_alpha 0.35 --height_adjust

# -- 去掉多线程：此行数据从 Table1 行1 读取，无需重跑
printf "%-45s\t%6s\t%8s\t%8s\n" \
    "  -- Multi-threading" \
    "$(grep 'Baseline' "${TABLE1}" | grep -oP '^[^\t]+\t\K[0-9.]+' | head -1 || echo N/A)" \
    "(->Table1)" "N/A" | tee -a "${TABLE3}"

# -- 去掉 FP16
WHAM_DETECT_INTERVAL=2 WHAM_INFER_INTERVAL=2 \
WHAM_STREAM_SEQ_LEN=8 WHAM_INPUT_SCALE=0.5 \
row3 "  -- FP16 (AMP)" "t3_02_no_fp16" --smooth_alpha 0.35 --height_adjust

# -- 去掉跳帧
WHAM_USE_AMP=1 WHAM_DETECT_INTERVAL=1 WHAM_INFER_INTERVAL=1 \
WHAM_STREAM_SEQ_LEN=8 WHAM_INPUT_SCALE=0.5 \
row3 "  -- Frame skipping (d=1)" "t3_03_no_skip" --smooth_alpha 0.35 --height_adjust

# -- 去掉短窗口 W=16
WHAM_USE_AMP=1 WHAM_DETECT_INTERVAL=2 WHAM_INFER_INTERVAL=2 \
WHAM_STREAM_SEQ_LEN=16 WHAM_INPUT_SCALE=0.5 \
row3 "  -- Reduced window (W=16)" "t3_04_no_w8" --smooth_alpha 0.35 --height_adjust

# -- 去掉缩放 s=1.0
WHAM_USE_AMP=1 WHAM_DETECT_INTERVAL=2 WHAM_INFER_INTERVAL=2 \
WHAM_STREAM_SEQ_LEN=8 WHAM_INPUT_SCALE=1.0 \
row3 "  -- Input scaling (s=1.0)" "t3_05_no_scale" --smooth_alpha 0.35 --height_adjust

echo "===========================================================================" | tee -a "${TABLE3}"
echo "Saved: ${TABLE3}"
echo ""
echo "All done. Logs in ${RESULTS_DIR}/runs/"
