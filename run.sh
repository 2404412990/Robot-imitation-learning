#!/usr/bin/env bash
set -euo pipefail

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

STREAM_CHUNK_SIZE=${STREAM_CHUNK_SIZE:-16}

SMOOTH_ALPHA=${SMOOTH_ALPHA:-0.35}
HEIGHT_ADJUST=${HEIGHT_ADJUST:-1}
ROOT_ORIGIN_OFFSET=${ROOT_ORIGIN_OFFSET:-1}
VIEWER_WARMUP_FRAMES=${VIEWER_WARMUP_FRAMES:-12}
POLL_INTERVAL=${POLL_INTERVAL:-0.05}
IDLE_TIMEOUT=${IDLE_TIMEOUT:-0}
DONE_GRACE_SEC=${DONE_GRACE_SEC:-2.0}

# GMR viewer camera controls.
CAMERA_FOLLOW=${CAMERA_FOLLOW:-1}
CAMERA_LOOKAT_HEIGHT_OFFSET=${CAMERA_LOOKAT_HEIGHT_OFFSET:-0.75}
CAMERA_ELEVATION=${CAMERA_ELEVATION:--5.0}
CAMERA_DISTANCE_SCALE=${CAMERA_DISTANCE_SCALE:-1.0}
CAMERA_AZIMUTH=${CAMERA_AZIMUTH:-}

mkdir -p "${OUTPUT_DIR}" "${STREAM_NPZ_DIR}" "$(dirname "${PKL_PATH}")" "$(dirname "${CSV_PATH}")" "$(dirname "${GMR_VIDEO_PATH}")"

# Clean stream artifacts before launching consumer to avoid mixing stale chunks from prior runs.
rm -f "${STREAM_NPZ_DIR}"/chunk_*.npz "${STREAM_NPZ_DIR}"/stream_done.flag

WHAM_CMD=(
	"${WHAM_PYTHON}" demo_stream_mt.py
	--video "${VIDEO}"
	--time "${TIME}"
	--output_dir "${OUTPUT_DIR}"
	--stream_npz_dir "${STREAM_NPZ_DIR}"
	--stream_chunk_size "${STREAM_CHUNK_SIZE}"
)

GMR_CMD=(
	"${GMR_PYTHON}" scripts/smplx_to_robot_stream.py
	--stream_npz_dir "${STREAM_NPZ_DIR}"
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
	GMR_CMD+=(--record_gmrvideo --rate_limit --video_path "${GMR_VIDEO_PATH}")
fi

echo "[E2E] Starting GMR consumer (${GMR_ENV}) with ${GMR_PYTHON} ..."
if [[ "${RECORD_GMRVIDEO}" == "1" && "${USE_XVFB_GMR}" == "1" ]]; then
	xvfb-run -a "${GMR_CMD[@]}" &
else
	"${GMR_CMD[@]}" &
fi
GMR_PID=$!

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
