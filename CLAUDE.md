# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unity ML-Agents project for humanoid robot imitation learning. The Unity side handles physics simulation, motion replay, and PPO training for humanoid robots (Unitree G1, Unitree H1). The Python side (see [Robot-imitation-learning/docs/CLAUDE.md](Robot-imitation-learning/docs/CLAUDE.md)) runs WHAM (video → 3D human motion) + GMR (human motion → robot joint IK).

## Architecture

### Unity Scenes

- `G1.unity` — Main scene with Unitree G1 (29 DOF). Contains UI for real-time WHAM capture + CSV replay.
- `G1Replay.unity` — G1 replay-only viewer scene.

### Interface & Registry Pattern

The codebase uses a clean abstraction layer in namespace `Gewu.Imitation`:

- [Scripts/IMimicAgent.cs](Scripts/IMimicAgent.cs) — Interface defining the contract for all imitation agents: `RobotKey`, `IsReplayMode`, `LoadCsvData()`, `ResetToReplay()`.
- [Scripts/MimicAgentRegistry.cs](Scripts/MimicAgentRegistry.cs) — Singleton registry that maps robot key strings to `IMimicAgent` instances. UI scripts (StartInput, Replay, Stop) look up agents through this registry, decoupling them from specific robot implementations.

### Agent Scripts (root Assets/Imitation/)

- [G1mimicAgent.cs](G1mimicAgent.cs) — **Primary** ML-Agents `Agent` for Unitree G1 (29 DOF). Implements `IMimicAgent`. Supports three modes:
  - **Training**: PPO with PD controller (Kp/Kd) to track reference pose. Reward = live + rot_error + pos_error + dof_error.
  - **Replay**: Kinematic teleporting (no physics) with CSV motion data.
  - **Live**: Real-time CSV ingestion from WHAM+GMR pipeline via `StartInput`.
- [H1mimicAgent.cs](H1mimicAgent.cs) — ML-Agents `Agent` for Unitree H1 (19 DOF). Implements `IMimicAgent`. Training clones 24 instances. Loads data from `StreamingAssets/h1_dataset/` subfolders containing `dof.csv`, `root_rot.csv`, `root_trans_offset.csv`.
- [G1mimic1Agent.cs](G1mimic1Agent.cs) — Older variant of G1 agent (does not implement `IMimicAgent`; uses direct float arrays).
- [G1mimicrealtime.cs](G1mimicrealtime.cs) — **Legacy** real-time G1 agent via UDP (class `G1mimicAgent_RealTime`). Does not implement `IMimicAgent`. Superseded by the WHAM+GMR pipeline via `StartInput`.
- [RobotController.cs](RobotController.cs) — MediaPipe-based teleoperation (class `G1Teleoperation`). Receives joint commands via UDP. Separate from the ML-Agents training pipeline.

### Scripts/ directory — UI & bridge utilities

- [StartInput.cs](Scripts/StartInput.cs) — **Key bridge script**: launches `run.ps1` (PowerShell, Windows) or `run.sh` (bash, Linux) as a subprocess, passes all config via environment variables (OUTPUT_ROOT, ROBOT, WHAM_USE_AMP, etc.), polls the output CSV (`csv/live_motion.csv`) on a coroutine, and feeds it to `G1mimicAgent` for real-time imitation. Handles process lifecycle including `taskkill /T` tree termination on Windows and `setsid` group kill on Linux.
- [Replay.cs](Scripts/Replay.cs) — UI button handler that selects a CSV from a dropdown and triggers replay mode on the agent (looked up via `MimicAgentRegistry`).
- [Stop.cs](Scripts/Stop.cs) — UI button to kill the WHAM+GMR pipeline and transition the agent to replay mode (reloads motion data from the dataset directory).
- [StreamReceiver.cs](Scripts/StreamReceiver.cs) — TCP server receiving JPEG frames from the Python pipeline for WHAM/GMR visualization in Unity.
- [FileBrowser.cs](Scripts/FileBrowser.cs) — Populates TMP_Dropdown with CSV files or folders. Supports `CsvFiles` and `Folders` modes.
- [Move.cs](Scripts/Move.cs) — FPS-style camera controller with mouse look and WASD movement.
- [FPSUiInteractor.cs](Scripts/FPSUiInteractor.cs) — FPS-style UI raycasting for clicking UI elements in 3D world space.
- [Reset.cs](Scripts/Reset.cs) — Placeholder (empty Start/Update).

### Data flow

```
Video/Camera → WHAM + GMR (Python, handle_wham_gmr.py, integrated pipeline)
    → live_motion.csv (36 floats/row)
    → StartInput coroutine polls CSV → G1mimicAgent.ApplyRealtimeCsvToAgent()
    → FixedUpdate applies joint targets via ArticulationBody PD drives
```

### CSV data format

Each CSV row = 36 floats: `[root_pos_x, root_pos_y, root_pos_z, root_rot_x, root_rot_y, root_rot_z, root_rot_w, dof_0, ..., dof_28]`

G1 motion types in `dataset/`: dance, walk, run, sprint, jump, fight, fallAndGetUp, fightAndSports (41 CSV files, multiple subjects).

### Key coordinate conventions

- WHAM outputs in Y-up (OpenGL); GMR/MuJoCo uses Z-up; Unity uses Y-up
- Root position mapping: `newPosition = Vector3(-pos[1], pos[2], pos[0])` — WHAM Y→Unity X (negated), WHAM Z→Unity Y
- Root rotation mapping: `newRotation = Quaternion(-rot[1], rot[2], rot[0], -rot[3])`
- Joint DOF values are in radians, converted to degrees: `dof[i] * 180f / π`

### ML-Agents configuration

[config.yaml](config.yaml) defines the `gewu` behavior with:
- **Trainer**: PPO, `max_steps: 200,000,000`
- **Hyperparameters**: `batch_size: 2048`, `buffer_size: 20480`, `time_horizon: 1000`, `beta: 0.005` (entropy)
- **Network**: 3 hidden layers × 512 units, `normalize: true`
- **Reward**: Extrinsic only, `gamma: 0.995`

### ONNX models

- `g1m1.onnx` — Trained G1 policy (29 DOF output)
- `h1-all-in-one.onnx` — Trained H1 policy (19 DOF output)

### H1 dataset (StreamingAssets/h1_dataset/)

H1 uses `.pkl` files (Python pickle) rather than CSV. Utility scripts:
- `trans.py` — Data transformation
- `viewnpy.py` — View `.npy` files
- `viewpkl.py` — View `.pkl` files

Motion data includes ACCAD martial arts kicks, dance (cha-cha, waltz), golf, and more.

## Environment Setup

### 1. Unity side

1. Install Unity Hub + Unity Editor 2022.3 (LTS)
2. Open `gewu/` directory as a Unity project
3. Dependencies auto-install from `Packages/manifest.json`:
   - `com.unity.ml-agents` (local package at `../com.unity.ml-agents`)
   - `com.unity.robotics.urdf-importer` (local package)
   - `com.unity.sentis` 2.1.0 (ONNX inference)
   - `com.unity.burst` 1.8.21
   - `com.unity.inputsystem` 1.14.0

### 2. Python `gewu` environment — ML-Agents training

```bash
conda create -n gewu python=3.10.12 -y
conda activate gewu
pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gewu.txt
mlagents-learn --help  # verify
```

### 3. Python `wham_gmr` environment — WHAM+GMR pipeline

**前置条件** (需手动安装):
- [CUDA Toolkit 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) — PyTorch3D/DPVO 编译需要
- [Visual Studio 2019 Community](https://visualstudio.microsoft.com/vs/older-downloads/) — C++ 编译器，**必须用 2019**

**自动安装脚本** (推荐):
```powershell
# 在 Anaconda PowerShell Prompt 中
cd Robot-imitation-learning
.\setup_env.ps1
```

**手动安装** (详见 [docs/INSTALL_WIN.md](Robot-imitation-learning/docs/INSTALL_WIN.md)):
```powershell
conda create -n wham_gmr python=3.10 -y
conda activate wham_gmr
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install setuptools==59.5.0
pip install --no-build-isolation mmcv==1.3.9
pip install -r requirements-wham_gmr.txt
pip install -v -e third-party/ViTPose
pip install -e .
```

**可选组件** (编译较难，可跳过):
- **PyTorch3D**: 需要 CUDA 11.3 + VS 2019 + CUB 1.11.0。仅用于 WHAM 可视化渲染。
- **DPVO**: 需要 CUDA 11.3 + VS 2019 + Eigen 3.4.0。仅用于全局 SLAM。

**下载资源文件**:
- 百度网盘: https://pan.baidu.com/s/1fVf2eA1OzdRv70M4gm2wSA?pwd=8pnu
- 解压到 `Robot-imitation-learning/` 下:
  - `checkpoints/` — WHAM 模型权重
  - `dataset/` — 训练/评估数据
  - `assets/body_models/` — SMPL-X 人体模型

### 4. Mac setup (alternative)

- Apple Silicon: Rosetta + Miniforge3 x86_64, Python 3.8, PyTorch 1.8.0, mlagents 0.28.0. See `macos-setup-Silicon.md`.
- Intel: Homebrew + Miniconda, Python 3.8, mlagents 0.28.0. See `macos-setup-Intelcore.md`.

## Training (ML-Agents PPO)

```bash
# From Assets/Imitation/ directory
mlagents-learn config.yaml --run-id=g1mimic --force

# Resume training
mlagents-learn config.yaml --run-id=g1mimic --resume

# View training metrics
tensorboard --logdir results --port 6006
# Open http://localhost:6006/
```

Training config: `config.yaml` (200M steps, PPO, batch 2048)

## Running the WHAM+GMR live pipeline

The pipeline streams video → 3D motion → Unity in real-time. `StartInput.cs` launches `run.ps1` and polls `csv/live_motion.csv`.

```powershell
# Example: Unitree H1 from drone video
cd Robot-imitation-learning
$env:OUTPUT_ROOT='output/h1_run'
$env:ROBOT='unitree_h1'
$env:RECORD_GMRVIDEO=1
$env:RECORD_WHAMVIDEO=1
$env:VIDEO='examples/drone_video.mp4'
./run.ps1
```

Key environment variables (set before `./run.ps1`):

| Variable | Default | Description |
|---|---|---|
| `VIDEO` | `examples/IMG_9732.mov` | Input video path; `0` = webcam |
| `ROBOT` | `unitree_g1` | Target robot type |
| `OUTPUT_ROOT` | `output/stream_demo` | Output directory |
| `WHAM_USE_AMP` | `0` | Half-precision WHAM inference |
| `WHAM_DETECT_INTERVAL` | `1` | Full detection every N frames |
| `WHAM_INFER_INTERVAL` | `1` | Full WHAM inference every N frames |
| `WHAM_STREAM_SEQ_LEN` | `16` | WHAM temporal window length |
| `WHAM_INPUT_SCALE` | `1.0` | Input scaling (0.1–1.0) |
| `GMR_TORCH_DEVICE` | `cpu` | GMR device (`cpu`/`cuda`/`auto`) |
| `E2E_WARMUP` | `1` | CUDA warmup on first run |
| `RECORD_WHAMVIDEO` | `1` | Save WHAM visualization |
| `RECORD_GMRVIDEO` | `1` | Render MuJoCo window |

Supported robots: `unitree_g1`, `unitree_h1`, `booster_t1`, `fourier_n1`, `engineai_pm01`, `kuavo_s45`, `openloong`, `tienkung`, and more (see `StartInput.cs`).

## Common commands

No Unity build commands — this is a Unity Editor project. Open in Unity, load `G1.unity` or `G1Replay.unity`, and press Play.

For the Python pipeline (WHAM + GMR), see [Robot-imitation-learning/docs/INSTALL_WIN.md](Robot-imitation-learning/docs/INSTALL_WIN.md) for detailed installation steps.

## Robot ArticulationBody details

- G1: 29 revolute joints, found by scanning all `ArticulationBody` components and filtering `jointType == RevoluteJoint`
- H1: 19 revolute joints
- PD gains for G1: `stiffness=180`, `damping=8`; for H1: `stiffness=2000`, `damping=200`
- `Time.fixedDeltaTime = 0.02` (50 Hz physics)
- Root body is `arts[0]`, teleported during replay, PD-force-driven during training

## Important edge cases

- **ArticulationBody cache size**: `SetJointPositions` requires exact DOF count matching. The G1 agent uses `SafeSetJointPositions`/`EnsureListSize` wrappers to trim/pad when the live CSV state changes size.
- **Rest pose capture**: `G1mimicAgent` snapshots rest positions into a fixed-size array at `Initialize()` time to prevent the list from mutating on subsequent `GetJointPositions` calls.
- **Immovable root timing**: Root starts immovable for first 3 physics steps, then releases to avoid initial settling artifacts.
- **Replay vs training physics**: In replay mode gravity is zeroed and the root is teleported each frame; in training mode the root is PD-force-driven.
- **bash process group handling**: `StartInput` wraps bash with `setsid` on Linux so `kill -9 -<pgid>` can terminate the entire WHAM+GMR pipeline tree.
