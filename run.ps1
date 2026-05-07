#Requires -Version 5.1
<#
.SYNOPSIS
  WHAM -> GMR end-to-end stream pipeline (PowerShell / Windows)

.DESCRIPTION
  PowerShell port of run.sh. Usage examples:
    $env:RECORD_GMRVIDEO = "1"; $env:RECORD_WHAMVIDEO = "0"; $env:VIDEO = "examples/IMG_9732.mov"; .\run.ps1
    $env:OUTPUT_ROOT = "outputs/run1"; $env:RECORD_GMRVIDEO = "0"; $env:VIDEO = "0"; .\run.ps1
    $env:VIDEO = "0"; $env:TIME = "10"; $env:RECORD_GMRVIDEO = "1"; $env:RECORD_WHAMVIDEO = "1"; .\run.ps1
#>

$ErrorActionPreference = "Stop"

# Ensure local packages are importable.
$env:PYTHONPATH = "$PWD$([IO.Path]::PathSeparator)$env:PYTHONPATH"

# =========================
# Default values (mirrors run.sh)
# =========================

function _env { param([string]$name, [string]$default)
    $v = [Environment]::GetEnvironmentVariable($name, "Process")
    if ($v) { return $v } else { return $default }
}

$WHAM_ENV     = _env WHAM_ENV     "wham_gmr"
$GMR_ENV      = _env GMR_ENV      "wham_gmr"

# Auto-detect Python path from conda env; fall back to PATH python.
$_autoPy = $null
# 1) CONDA_PREFIX (set when conda env is activated)
if ($env:CONDA_PREFIX) {
    $p = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path $p) { $_autoPy = $p }
}
# 2) Scan common conda install locations
if (-not $_autoPy) {
    foreach ($_base in @("$env:USERPROFILE\.conda", "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3")) {
        $p = Join-Path $_base "envs\$WHAM_ENV\python.exe"
        if (Test-Path $p) { $_autoPy = $p; break }
    }
}
# 3) Last resort: use python from PATH
if (-not $_autoPy) { $_autoPy = "python" }
$WHAM_PYTHON  = _env WHAM_PYTHON $_autoPy
$GMR_PYTHON   = _env GMR_PYTHON   $WHAM_PYTHON

$VIDEO        = _env VIDEO        "examples/IMG_9732.mov"
$TIME         = _env TIME         "0"
$ROBOT        = _env ROBOT        "unitree_g1"
$ROBOT_PATH   = _env ROBOT_PATH   ""

# ROBOT_PATH supports both robot key and custom XML path.
$CUSTOM_ROBOT_XML = ""
if ($ROBOT_PATH) {
    if ((Test-Path $ROBOT_PATH) -or $ROBOT_PATH.EndsWith(".xml")) {
        $CUSTOM_ROBOT_XML = $ROBOT_PATH
    } else {
        $ROBOT  = $ROBOT_PATH
    }
}

# Backward compatibility.
$LEGACY_RECORD_VIDEO  = _env RECORD_VIDEO   "1"
$RECORD_WHAMVIDEO     = _env RECORD_WHAMVIDEO $LEGACY_RECORD_VIDEO
$RECORD_GMRVIDEO      = _env RECORD_GMRVIDEO  $LEGACY_RECORD_VIDEO
$USE_XVFB_GMR         = _env USE_XVFB_GMR     "0"

$OUTPUT_ROOT    = _env OUTPUT_ROOT ""
if ($OUTPUT_ROOT) {
    $OUTPUT_DIR      = _env OUTPUT_DIR     "$OUTPUT_ROOT/stream_demo"
    $STREAM_NPZ_DIR  = _env STREAM_NPZ_DIR "$OUTPUT_DIR/npz_stream"
    $PKL_PATH        = _env PKL_PATH       "$OUTPUT_ROOT/pkl/my_motion.pkl"
    $CSV_PATH        = _env CSV_PATH       "$OUTPUT_ROOT/csv/live_motion.csv"
    $GMR_VIDEO_PATH  = _env GMR_VIDEO_PATH "$OUTPUT_ROOT/video/live_stream_robot.mp4"
} else {
    $OUTPUT_DIR      = _env OUTPUT_DIR     "output/stream_demo"
    $STREAM_NPZ_DIR  = _env STREAM_NPZ_DIR "$OUTPUT_DIR/npz_stream"
    $PKL_PATH        = _env PKL_PATH       "pkl_outputs/my_motion.pkl"
    $CSV_PATH        = _env CSV_PATH       "pkl_outputs/csv/live_motion.csv"
    $GMR_VIDEO_PATH  = _env GMR_VIDEO_PATH "videos/live_stream_robot.mp4"
}

$SMOOTH_ALPHA          = _env SMOOTH_ALPHA          "0.35"
$HEIGHT_ADJUST         = _env HEIGHT_ADJUST         "1"
$ROOT_ORIGIN_OFFSET    = _env ROOT_ORIGIN_OFFSET    "0"
$VIEWER_WARMUP_FRAMES  = _env VIEWER_WARMUP_FRAMES  "12"
$POLL_INTERVAL         = _env POLL_INTERVAL         "0.05"
$IDLE_TIMEOUT          = _env IDLE_TIMEOUT          "0"
$DONE_GRACE_SEC        = _env DONE_GRACE_SEC        "2.0"

# WHAM performance controls.
$WHAM_USE_AMP          = _env WHAM_USE_AMP          "0"
$WHAM_DETECT_INTERVAL  = _env WHAM_DETECT_INTERVAL  "1"
$WHAM_INFER_INTERVAL   = _env WHAM_INFER_INTERVAL   "1"
$WHAM_STREAM_SEQ_LEN   = _env WHAM_STREAM_SEQ_LEN   "16"
$WHAM_INPUT_SCALE      = _env WHAM_INPUT_SCALE      "1.0"
$env:WHAM_USE_AMP         = $WHAM_USE_AMP
$env:WHAM_DETECT_INTERVAL = $WHAM_DETECT_INTERVAL
$env:WHAM_INFER_INTERVAL  = $WHAM_INFER_INTERVAL
$env:WHAM_STREAM_SEQ_LEN  = $WHAM_STREAM_SEQ_LEN
$env:WHAM_INPUT_SCALE     = $WHAM_INPUT_SCALE

$GMR_TORCH_DEVICE                     = _env GMR_TORCH_DEVICE                     "cpu"
$GMR_VIEWER_READY_TIMEOUT_SEC         = _env GMR_VIEWER_READY_TIMEOUT_SEC         "8"
$GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC   = _env GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC   "10"

# One-time warmup.
$E2E_WARMUP           = _env E2E_WARMUP           "1"
$E2E_WARMUP_FORCE     = _env E2E_WARMUP_FORCE     "0"
$E2E_WARMUP_ONCE      = _env E2E_WARMUP_ONCE      "1"
$E2E_WARMUP_CACHE_ROOT= _env E2E_WARMUP_CACHE_ROOT "$PWD\.cache\wham-gmr"

# GMR viewer camera controls.
$CAMERA_FOLLOW               = _env CAMERA_FOLLOW               "0"
$CAMERA_LOOKAT_HEIGHT_OFFSET = _env CAMERA_LOOKAT_HEIGHT_OFFSET "0.45"
$CAMERA_ELEVATION            = _env CAMERA_ELEVATION            "12.0"
$CAMERA_DISTANCE_SCALE       = _env CAMERA_DISTANCE_SCALE       "0.85"
$CAMERA_AZIMUTH              = _env CAMERA_AZIMUTH              ""

# Create output directories.
foreach ($d in @(
    $OUTPUT_DIR,
    (Split-Path -Parent $PKL_PATH),
    (Split-Path -Parent $CSV_PATH),
    (Split-Path -Parent $GMR_VIDEO_PATH)
)) {
    if ($d) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

$GMR_READY_FLAG   = _env GMR_READY_FLAG   "$STREAM_NPZ_DIR/gmr_ready.flag"
$STREAM_TAIL_PATH = _env STREAM_TAIL_PATH "$STREAM_NPZ_DIR/stream_tail.pkl"
$READY_TIMEOUT_SEC = 60
$STARTUP_DELAY_SEC = 0.5

# Clean stale stream artifacts.
Remove-Item -Force -ErrorAction SilentlyContinue "$STREAM_NPZ_DIR/chunk_*.npz"
Remove-Item -Force -ErrorAction SilentlyContinue "$STREAM_NPZ_DIR/stream_done.flag"
Remove-Item -Force -ErrorAction SilentlyContinue $GMR_READY_FLAG
Remove-Item -Force -ErrorAction SilentlyContinue $STREAM_TAIL_PATH

$_display = if ($env:DISPLAY) { $env:DISPLAY } else { '<unset>' }
Write-Host "[E2E] Render flags: RECORD_WHAMVIDEO=$RECORD_WHAMVIDEO RECORD_GMRVIDEO=$RECORD_GMRVIDEO USE_XVFB_GMR=$USE_XVFB_GMR DISPLAY=$_display"
Write-Host "[E2E] Camera params: follow=$CAMERA_FOLLOW lookat_h=$CAMERA_LOOKAT_HEIGHT_OFFSET elev=$CAMERA_ELEVATION dist_scale=$CAMERA_DISTANCE_SCALE azimuth=$(if ($CAMERA_AZIMUTH) { $CAMERA_AZIMUTH } else { '<auto>' })"
Write-Host "[E2E] WHAM perf params: amp=$WHAM_USE_AMP detect_interval=$WHAM_DETECT_INTERVAL infer_interval=$WHAM_INFER_INTERVAL seq_len=$WHAM_STREAM_SEQ_LEN input_scale=$WHAM_INPUT_SCALE"
Write-Host "[E2E] WHAM stream mode=tail (fixed)"
Write-Host "[E2E] GMR torch_device=$GMR_TORCH_DEVICE"
Write-Host "[E2E] GMR viewer mode: async thread + low-latency fixed profile (ready_timeout=$GMR_VIEWER_READY_TIMEOUT_SEC s, join_timeout=$GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC s)"
Write-Host "[E2E] Warmup: enabled=$E2E_WARMUP once=$E2E_WARMUP_ONCE force=$E2E_WARMUP_FORCE"

if ($RECORD_GMRVIDEO -eq "1" -and $USE_XVFB_GMR -eq "1") {
    Write-Host "[E2E] GMR is running with xvfb; MuJoCo window will not appear on physical screen."
    Write-Warning "[E2E] xvfb-run is not available on Windows. Falling back to direct rendering."
}
if ($RECORD_GMRVIDEO -eq "1" -and -not $env:DISPLAY) {
    Write-Host "[E2E] Warning: DISPLAY is empty; MuJoCo window may not appear."
}

# ---- warmup helpers ----

function Invoke-EnvWarmup {
    param([string]$Name, [string]$PyBin)
    if (-not (Test-Path $PyBin -PathType Leaf)) {
        Write-Host "[E2E] Warmup warning: $Name python not found: $PyBin"
        return
    }

    $tmpFile = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.IO.Path]::GetRandomFileName() + ".py")
    try {
        @'
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
'@ | Out-File -FilePath $tmpFile -Encoding utf8

        $result = & $PyBin $tmpFile 2>&1
        $exitCode = $LASTEXITCODE
        Write-Host $result
        if ($exitCode -ne 0) {
            Write-Host "[E2E] Warmup warning: $Name warmup command failed (ignored)."
        }
    } finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $tmpFile
    }
}

function Invoke-E2EWarmupIfNeeded {
    if ($E2E_WARMUP -ne "1") { return }

    $warmupMarker = Join-Path $E2E_WARMUP_CACHE_ROOT "warmup_${WHAM_ENV}_${GMR_ENV}.done"

    if ($E2E_WARMUP_ONCE -eq "1" -and $E2E_WARMUP_FORCE -ne "1" -and (Test-Path $warmupMarker)) {
        Write-Host "[E2E] Warmup already done before, skip ($warmupMarker)."
        return
    }

    Write-Host "[E2E] Running warmup for $WHAM_ENV / $GMR_ENV ..."
    Invoke-EnvWarmup "WHAM" $WHAM_PYTHON
    Invoke-EnvWarmup "GMR"  $GMR_PYTHON

    if ($E2E_WARMUP_ONCE -eq "1") {
        New-Item -ItemType Directory -Force -Path $E2E_WARMUP_CACHE_ROOT | Out-Null
        [DateTimeOffset]::Now.ToUnixTimeSeconds() | Out-File -FilePath $warmupMarker -NoNewline
        Write-Host "[E2E] Warmup marker written: $warmupMarker"
    }
}

# ---- run warmup ----
Invoke-E2EWarmupIfNeeded

# ---- build integrated command ----

Write-Host "[E2E] Starting integrated WHAM + GMR producer/consumer with $WHAM_PYTHON ..."

$INTEGRATED_CMD = @(
    $WHAM_PYTHON, "handle_wham_gmr.py",
    "--video", $VIDEO,
    "--time", $TIME,
    "--output_dir", $OUTPUT_DIR,
    "--robot", $ROBOT,
    "--coord_fix", "yup_to_zup",
    "--save_path", $PKL_PATH,
    "--csv_path", $CSV_PATH,
    "--smooth_alpha", $SMOOTH_ALPHA,
    "--viewer_warmup_frames", $VIEWER_WARMUP_FRAMES,
    "--camera_lookat_height_offset", $CAMERA_LOOKAT_HEIGHT_OFFSET,
    "--camera_elevation", $CAMERA_ELEVATION,
    "--camera_distance_scale", $CAMERA_DISTANCE_SCALE,
    "--poll_interval", $POLL_INTERVAL,
    "--idle_timeout", $IDLE_TIMEOUT,
    "--done_grace_sec", $DONE_GRACE_SEC,
    "--torch_device", $GMR_TORCH_DEVICE
)

if ($CUSTOM_ROBOT_XML) {
    $INTEGRATED_CMD += @("--robot_path", $CUSTOM_ROBOT_XML)
}

if ($CAMERA_FOLLOW -eq "1") {
    $INTEGRATED_CMD += "--camera_follow"
} else {
    $INTEGRATED_CMD += "--no-camera_follow"
}

if ($CAMERA_AZIMUTH) {
    $INTEGRATED_CMD += @("--camera_azimuth", $CAMERA_AZIMUTH)
}

$INTEGRATED_CMD += @("--viewer_ready_timeout_sec", $GMR_VIEWER_READY_TIMEOUT_SEC)
$INTEGRATED_CMD += @("--viewer_thread_join_timeout_sec", $GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC)

if ($HEIGHT_ADJUST -eq "1") {
    $INTEGRATED_CMD += "--height_adjust"
} else {
    $INTEGRATED_CMD += "--no-height_adjust"
}

if ($ROOT_ORIGIN_OFFSET -eq "1") {
    $INTEGRATED_CMD += "--root_origin_offset"
} else {
    $INTEGRATED_CMD += "--no-root_origin_offset"
}

if ($RECORD_WHAMVIDEO -eq "1") {
    $INTEGRATED_CMD += "--record_whamvideo"
}

if ($RECORD_GMRVIDEO -eq "1") {
    $INTEGRATED_CMD += @("--record_gmrvideo", "--video_path", $GMR_VIDEO_PATH)
}

# xvfb-run is unavailable on Windows — always run directly.
$exe = $INTEGRATED_CMD[0]
$cmdArgs = if ($INTEGRATED_CMD.Length -gt 1) { $INTEGRATED_CMD[1..($INTEGRATED_CMD.Length - 1)] } else { @() }
& $exe $cmdArgs

Write-Host "[E2E] Done."
Write-Host "  CSV: $CSV_PATH"
Write-Host "  PKL: $PKL_PATH"
if ($RECORD_GMRVIDEO -eq "1") {
    Write-Host "  GMR video: $GMR_VIDEO_PATH"
}
