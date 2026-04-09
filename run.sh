#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
	echo "[E2E] This script requires bash. Please run: bash run.sh"
	exit 2
fi

set -euo pipefail

# Ensure local packages are importable even when scripts are launched by path.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# =========================
# End-to-end WHAM -> GMR stream pipeline
# =========================
# Usage examples:
#   RECORD_GMRVIDEO=1 RECORD_WHAMVIDEO=0 VIDEO=examples/IMG_9732.mov bash run.sh
#   OUTPUT_ROOT=outputs/run1 RECORD_GMRVIDEO=0 VIDEO=0 bash run.sh
#   VIDEO=0 TIME=10 RECORD_GMRVIDEO=1 RECORD_WHAMVIDEO=1 bash run.sh

WHAM_ENV=${WHAM_ENV:-wham}
GMR_ENV=${GMR_ENV:-gmr}
WHAM_PYTHON=${WHAM_PYTHON:-/home/shaochang/anaconda3/envs/${WHAM_ENV}/bin/python}
GMR_PYTHON=${GMR_PYTHON:-/home/shaochang/anaconda3/envs/${GMR_ENV}/bin/python}

VIDEO=${VIDEO:-examples/IMG_9732.mov}
TIME=${TIME:-0}
ROBOT=${ROBOT:-unitree_g1}
ROBOT_PATH=${ROBOT_PATH:-}

# ROBOT_PATH supports both robot key (same choices as --robot) and custom XML path.
if [[ -n "${ROBOT_PATH}" ]]; then
	if [[ -f "${ROBOT_PATH}" || "${ROBOT_PATH}" == *.xml ]]; then
		CUSTOM_ROBOT_XML="${ROBOT_PATH}"
	else
		ROBOT="${ROBOT_PATH}"
		CUSTOM_ROBOT_XML=""
	fi
else
	CUSTOM_ROBOT_XML=""
fi

# Backward compatibility: old RECORD_VIDEO controls both sides when specific flags are not set.
LEGACY_RECORD_VIDEO=${RECORD_VIDEO:-1}
RECORD_WHAMVIDEO=${RECORD_WHAMVIDEO:-${LEGACY_RECORD_VIDEO}}
RECORD_GMRVIDEO=${RECORD_GMRVIDEO:-${LEGACY_RECORD_VIDEO}}
USE_XVFB_GMR=${USE_XVFB_GMR:-0}

OUTPUT_ROOT=${OUTPUT_ROOT:-}
if [[ -n "${OUTPUT_ROOT}" ]]; then
	OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/stream_demo}
	STREAM_NPZ_DIR=${STREAM_NPZ_DIR:-${OUTPUT_DIR}/npz_stream}
	PKL_PATH=${PKL_PATH:-${OUTPUT_ROOT}/pkl/my_motion.pkl}
	CSV_PATH=${CSV_PATH:-${OUTPUT_ROOT}/csv/live_motion.csv}
	GMR_VIDEO_PATH=${GMR_VIDEO_PATH:-${OUTPUT_ROOT}/video/live_stream_robot.mp4}
else
	OUTPUT_DIR=${OUTPUT_DIR:-output/stream_demo}
	STREAM_NPZ_DIR=${STREAM_NPZ_DIR:-${OUTPUT_DIR}/npz_stream}
	PKL_PATH=${PKL_PATH:-pkl_outputs/my_motion.pkl}
	CSV_PATH=${CSV_PATH:-pkl_outputs/csv/live_motion.csv}
	GMR_VIDEO_PATH=${GMR_VIDEO_PATH:-videos/live_stream_robot.mp4}
fi

SMOOTH_ALPHA=${SMOOTH_ALPHA:-0.35}
HEIGHT_ADJUST=${HEIGHT_ADJUST:-1}
ROOT_ORIGIN_OFFSET=${ROOT_ORIGIN_OFFSET:-0}
VIEWER_WARMUP_FRAMES=${VIEWER_WARMUP_FRAMES:-12}
POLL_INTERVAL=${POLL_INTERVAL:-0.05}
IDLE_TIMEOUT=${IDLE_TIMEOUT:-0}
DONE_GRACE_SEC=${DONE_GRACE_SEC:-2.0}

# WHAM performance controls (can be overridden via env before running run.sh).
WHAM_USE_AMP=${WHAM_USE_AMP:-0}
WHAM_DETECT_INTERVAL=${WHAM_DETECT_INTERVAL:-1}
WHAM_INFER_INTERVAL=${WHAM_INFER_INTERVAL:-1}
WHAM_STREAM_SEQ_LEN=${WHAM_STREAM_SEQ_LEN:-16}
WHAM_INPUT_SCALE=${WHAM_INPUT_SCALE:-1.0}
export WHAM_USE_AMP WHAM_DETECT_INTERVAL WHAM_INFER_INTERVAL WHAM_STREAM_SEQ_LEN WHAM_INPUT_SCALE

GMR_TORCH_DEVICE=${GMR_TORCH_DEVICE:-cpu}

GMR_VIEWER_READY_TIMEOUT_SEC=${GMR_VIEWER_READY_TIMEOUT_SEC:-8}
GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC=${GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC:-10}

# One-time warmup to reduce first-run CUDA/PyTorch cold-start stalls.
E2E_WARMUP=${E2E_WARMUP:-1}
E2E_WARMUP_FORCE=${E2E_WARMUP_FORCE:-0}
E2E_WARMUP_ONCE=${E2E_WARMUP_ONCE:-1}
E2E_WARMUP_CACHE_ROOT=${E2E_WARMUP_CACHE_ROOT:-${PWD}/.cache/wham-gmr}

# GMR viewer camera controls.
CAMERA_FOLLOW=${CAMERA_FOLLOW:-0}
# Default to a lower, slightly top-down and closer camera framing.
CAMERA_LOOKAT_HEIGHT_OFFSET=${CAMERA_LOOKAT_HEIGHT_OFFSET:-0.45}
CAMERA_ELEVATION=${CAMERA_ELEVATION:-12.0}
CAMERA_DISTANCE_SCALE=${CAMERA_DISTANCE_SCALE:-0.85}
CAMERA_AZIMUTH=${CAMERA_AZIMUTH:-}

mkdir -p "${OUTPUT_DIR}" "${STREAM_NPZ_DIR}" "$(dirname "${PKL_PATH}")" "$(dirname "${CSV_PATH}")" "$(dirname "${GMR_VIDEO_PATH}")"

GMR_READY_FLAG=${GMR_READY_FLAG:-${STREAM_NPZ_DIR}/gmr_ready.flag}
STREAM_TAIL_PATH=${STREAM_TAIL_PATH:-${STREAM_NPZ_DIR}/stream_tail.pkl}
READY_TIMEOUT_SEC=60
STARTUP_DELAY_SEC=0.5

# Clean stream artifacts before launching consumer to avoid mixing stale chunks from prior runs.
rm -f "${STREAM_NPZ_DIR}"/chunk_*.npz "${STREAM_NPZ_DIR}"/stream_done.flag "${GMR_READY_FLAG}" "${STREAM_TAIL_PATH}"

WHAM_CMD=(
	"${WHAM_PYTHON}" demo_stream_mt.py
	--video "${VIDEO}"
	--time "${TIME}"
	--output_dir "${OUTPUT_DIR}"
	--stream_npz_dir "${STREAM_NPZ_DIR}"
)

GMR_CMD=(
	"${GMR_PYTHON}" scripts/smplx_to_robot_stream.py
	--stream_npz_dir "${STREAM_NPZ_DIR}"
	--stream_mode tail
	--stream_tail_path "${STREAM_TAIL_PATH}"
	--torch_device "${GMR_TORCH_DEVICE}"
	--ready_flag_path "${GMR_READY_FLAG}"
	--robot "${ROBOT}"
	--coord_fix yup_to_zup
	--save_path "${PKL_PATH}"
	--csv_path "${CSV_PATH}"
	--smooth_alpha "${SMOOTH_ALPHA}"
	--viewer_warmup_frames "${VIEWER_WARMUP_FRAMES}"
	--camera_lookat_height_offset "${CAMERA_LOOKAT_HEIGHT_OFFSET}"
	--camera_elevation "${CAMERA_ELEVATION}"
	--camera_distance_scale "${CAMERA_DISTANCE_SCALE}"
	--poll_interval "${POLL_INTERVAL}"
	--idle_timeout "${IDLE_TIMEOUT}"
	--done_grace_sec "${DONE_GRACE_SEC}"
)

if [[ -n "${CUSTOM_ROBOT_XML}" ]]; then
	GMR_CMD+=(--robot_path "${CUSTOM_ROBOT_XML}")
fi

if [[ "${CAMERA_FOLLOW}" == "1" ]]; then
	GMR_CMD+=(--camera_follow)
else
	GMR_CMD+=(--no-camera_follow)
fi

if [[ -n "${CAMERA_AZIMUTH}" ]]; then
	GMR_CMD+=(--camera_azimuth "${CAMERA_AZIMUTH}")
fi

GMR_CMD+=(--viewer_ready_timeout_sec "${GMR_VIEWER_READY_TIMEOUT_SEC}")
GMR_CMD+=(--viewer_thread_join_timeout_sec "${GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC}")

if [[ "${HEIGHT_ADJUST}" == "1" ]]; then
	GMR_CMD+=(--height_adjust)
else
	GMR_CMD+=(--no-height_adjust)
fi

if [[ "${ROOT_ORIGIN_OFFSET}" == "1" ]]; then
	GMR_CMD+=(--root_origin_offset)
else
	GMR_CMD+=(--no-root_origin_offset)
fi

if [[ "${RECORD_WHAMVIDEO}" == "1" ]]; then
	WHAM_CMD+=(--record_whamvideo)
fi

if [[ "${RECORD_GMRVIDEO}" == "1" ]]; then
	GMR_CMD+=(--record_gmrvideo --video_path "${GMR_VIDEO_PATH}")
fi

echo "[E2E] Render flags: RECORD_WHAMVIDEO=${RECORD_WHAMVIDEO} RECORD_GMRVIDEO=${RECORD_GMRVIDEO} USE_XVFB_GMR=${USE_XVFB_GMR} DISPLAY=${DISPLAY:-<unset>}"
echo "[E2E] Camera params: follow=${CAMERA_FOLLOW} lookat_h=${CAMERA_LOOKAT_HEIGHT_OFFSET} elev=${CAMERA_ELEVATION} dist_scale=${CAMERA_DISTANCE_SCALE} azimuth=${CAMERA_AZIMUTH:-<auto>}"
echo "[E2E] WHAM perf params: amp=${WHAM_USE_AMP} detect_interval=${WHAM_DETECT_INTERVAL} infer_interval=${WHAM_INFER_INTERVAL} seq_len=${WHAM_STREAM_SEQ_LEN} input_scale=${WHAM_INPUT_SCALE}"
echo "[E2E] WHAM stream mode=tail (fixed)"
echo "[E2E] GMR torch_device=${GMR_TORCH_DEVICE}"
echo "[E2E] GMR viewer mode: async thread + low-latency fixed profile (ready_timeout=${GMR_VIEWER_READY_TIMEOUT_SEC}s, join_timeout=${GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC}s)"
echo "[E2E] Warmup: enabled=${E2E_WARMUP} once=${E2E_WARMUP_ONCE} force=${E2E_WARMUP_FORCE}"
if [[ "${RECORD_GMRVIDEO}" == "1" && "${USE_XVFB_GMR}" == "1" ]]; then
	echo "[E2E] GMR is running with xvfb; MuJoCo window will not appear on physical screen."
elif [[ "${RECORD_GMRVIDEO}" == "1" && -z "${DISPLAY:-}" ]]; then
	echo "[E2E] Warning: DISPLAY is empty; MuJoCo window may not appear."
fi

run_env_warmup() {
	local name="$1"
	local py_bin="$2"
	if [[ ! -x "${py_bin}" ]]; then
		echo "[E2E] Warmup warning: ${name} python not executable: ${py_bin}"
		return 0
	fi

	if ! "${py_bin}" - <<'PY'
import time

try:
    import numpy as np
    import torch

    t0 = time.time()
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        x = torch.randn(512, 512, device=dev)
        y = torch.randn(512, 512, device=dev)
        _ = x @ y
        w = torch.randn(1, 3, 128, 128, device=dev)
        k = torch.randn(16, 3, 3, 3, device=dev)
        _ = torch.nn.functional.conv2d(w, k, padding=1)
        torch.cuda.synchronize()
    else:
        x = np.random.randn(256, 256).astype(np.float32)
        _ = x @ x.T

    dt = time.time() - t0
    print(f"[E2E] Warmup python ok in {dt:.3f}s")
except Exception as e:
    print(f"[E2E] Warmup warning: {e}")
PY
	then
		echo "[E2E] Warmup warning: ${name} warmup command failed (ignored)."
	fi
}

run_e2e_warmup_if_needed() {
	if [[ "${E2E_WARMUP}" != "1" ]]; then
		return 0
	fi

	local warmup_cache_root="${E2E_WARMUP_CACHE_ROOT}"
	local warmup_marker="${warmup_cache_root}/warmup_${WHAM_ENV}_${GMR_ENV}.done"

	if [[ "${E2E_WARMUP_ONCE}" == "1" && "${E2E_WARMUP_FORCE}" != "1" && -f "${warmup_marker}" ]]; then
		echo "[E2E] Warmup already done before, skip (${warmup_marker})."
		return 0
	fi

	echo "[E2E] Running warmup for ${WHAM_ENV}/${GMR_ENV} ..."
	run_env_warmup "WHAM" "${WHAM_PYTHON}"
	run_env_warmup "GMR" "${GMR_PYTHON}"

	if [[ "${E2E_WARMUP_ONCE}" == "1" ]]; then
		mkdir -p "${warmup_cache_root}"
		date +%s > "${warmup_marker}" || true
		echo "[E2E] Warmup marker written: ${warmup_marker}"
	fi
}

run_e2e_warmup_if_needed

echo "[E2E] Starting GMR consumer (${GMR_ENV}) with ${GMR_PYTHON} ..."
if [[ "${RECORD_GMRVIDEO}" == "1" && "${USE_XVFB_GMR}" == "1" ]]; then
	xvfb-run -a "${GMR_CMD[@]}" &
else
	"${GMR_CMD[@]}" &
fi
GMR_PID=$!

# Fast-fail if consumer exits immediately (e.g., import errors) to avoid wasting
# time running WHAM producer while GMR is already dead.
GMR_START_WAIT_SEC=${GMR_START_WAIT_SEC:-2}
sleep "${GMR_START_WAIT_SEC}"
if ! kill -0 "${GMR_PID}" 2>/dev/null; then
	echo "[E2E] GMR consumer exited during startup. Check the error above."
	wait "${GMR_PID}" || true
	exit 1
fi

# Wait for consumer readiness marker so producer starts only after GMR is ready.
echo "[E2E] Waiting GMR ready flag: ${GMR_READY_FLAG} (timeout=${READY_TIMEOUT_SEC}s) ..."
ready_start_ts=$(date +%s)
while [[ ! -f "${GMR_READY_FLAG}" ]]; do
	if ! kill -0 "${GMR_PID}" 2>/dev/null; then
		echo "[E2E] GMR consumer exited before ready flag was created."
		wait "${GMR_PID}" || true
		exit 1
	fi
	now_ts=$(date +%s)
	elapsed=$(( now_ts - ready_start_ts ))
	if [[ ${elapsed} -ge ${READY_TIMEOUT_SEC} ]]; then
		echo "[E2E] Warning: timed out waiting for ready flag; continue startup."
		break
	fi
	sleep 0.1
done

if [[ -f "${GMR_READY_FLAG}" ]]; then
	echo "[E2E] GMR ready flag detected."
fi

if [[ "${STARTUP_DELAY_SEC}" != "0" ]]; then
	echo "[E2E] Additional startup delay before WHAM: ${STARTUP_DELAY_SEC}s"
	sleep "${STARTUP_DELAY_SEC}"
fi

cleanup() {
	if [[ -n "${GMR_PID:-}" ]] && kill -0 "${GMR_PID}" 2>/dev/null; then
		echo "[E2E] Stopping GMR consumer (pid=${GMR_PID}) ..."
		kill "${GMR_PID}" 2>/dev/null || true
		wait "${GMR_PID}" 2>/dev/null || true
	fi
}
trap cleanup EXIT INT TERM

echo "[E2E] Starting WHAM producer (${WHAM_ENV}) with ${WHAM_PYTHON} ..."
"${WHAM_CMD[@]}"

echo "[E2E] Waiting GMR consumer to finish ..."
wait "${GMR_PID}"
trap - EXIT INT TERM

echo "[E2E] Done."
echo "  CSV: ${CSV_PATH}"
echo "  PKL: ${PKL_PATH}"
if [[ "${RECORD_GMRVIDEO}" == "1" ]]; then
	echo "  GMR video: ${GMR_VIDEO_PATH}"
fi
