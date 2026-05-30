# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unity ML-Agents project for humanoid robot imitation learning. The Unity side handles physics simulation, motion replay, and PPO training for humanoid robots (Unitree G1, Unitree H1). The Python side (see [Robot-imitation-learning/docs/CLAUDE.md](Robot-imitation-learning/docs/CLAUDE.md)) runs WHAM (video → 3D human motion) + GMR (human motion → robot joint IK).

## Architecture

### Unity Scenes

- `G1.unity` — Main scene with Unitree G1 (29 DOF). Contains UI for real-time WHAM capture + CSV replay.
- `G1Replay.unity` — G1 replay-only viewer scene.

### Core Scripts (root Assets/Imitation/)

- [G1mimicAgent.cs](G1mimicAgent.cs) — ML-Agents `Agent` for Unitree G1 (29 DOF). Supports three modes:
  - **Training**: PPO with PD controller (Kp/Kd) to track reference pose. Reward = live + rot_error + pos_error + dof_error.
  - **Replay**: Kinematic teleporting (no physics) with CSV motion data.
  - **Live**: Real-time CSV ingestion from WHAM+GMR pipeline via `StartInput`.
- [G1mimic1Agent.cs](G1mimic1Agent.cs) — Variant of G1mimicAgent (similar structure, different robot configuration).
- [H1mimicAgent.cs](H1mimicAgent.cs) — ML-Agents `Agent` for Unitree H1 (19 DOF). Training clones 24 instances. Loads data from `StreamingAssets/h1_dataset/` subfolders containing `dof.csv`, `root_rot.csv`, `root_trans_offset.csv`.

### Scripts/ directory — UI & bridge utilities

- [StartInput.cs](Scripts/StartInput.cs) — **Key bridge script**: launches `run.ps1` (PowerShell, Windows) or `run.sh` (bash, Linux) as a subprocess, passes all config via environment variables (OUTPUT_ROOT, ROBOT, WHAM_USE_AMP, etc.), polls the output CSV (`csv/live_motion.csv`) on a coroutine, and feeds it to `G1mimicAgent` for real-time imitation. Handles process lifecycle including `taskkill /T` tree termination on Windows and `setsid` group kill on Linux.
- [Replay.cs](Scripts/Replay.cs) — UI button handler that selects a CSV from a dropdown and triggers replay mode on `G1mimicAgent`.
- [Stop.cs](Scripts/Stop.cs) — UI button to kill the WHAM+GMR pipeline and transition the agent to replay mode (reloads motion data from the dataset directory).
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

## Common commands

No Unity build commands — this is a Unity Editor project. Open in Unity, load `G1.unity` or `G1Replay.unity`, and press Play.

For the Python pipeline (WHAM + GMR), see [Robot-imitation-learning/docs/CLAUDE.md](Robot-imitation-learning/docs/CLAUDE.md) for `run.sh` usage and environment variables.

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
