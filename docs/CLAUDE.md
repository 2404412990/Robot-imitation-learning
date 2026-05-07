# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

WHAM + GMR 集成系统：从视频/摄像头输入中提取 3D 人体运动（WHAM），并通过逆运动学（IK）实时重定向到人形机器人（GMR）。

- **WHAM**（Whole-body Humanoid Articulation Model）：基于视频的单目 3D 人体运动估计，输出 SMPL-X 参数
- **GMR**（General Motion Retargeting）：将人体运动重定向到 9+ 种人形机器人的 IK 求解器
- **集成管线**（`handle_wham_gmr.py` + `run.sh`）：将两者串联为实时流式处理链路

## 环境与安装

此项目使用单一 conda 环境 `wham_gmr`（`handle_wham_gmr.py` 在单进程中运行 WHAM 和 GMR）：

- **Python 3.10** + **PyTorch 1.11.0** + **CUDA 11.3**

详细安装步骤见 [docs/INSTALL.md](docs/INSTALL.md)。关键依赖：

- PyTorch 1.11.0 + CUDA 11.3
- ViTPose（第三方人体关键点检测）
- DPVO（可选，用于全局 SLAM 相机运动估计）
- SMPL-X 人体模型（需下载放置于 `assets/body_models/smplx/`）
- mink（IK 求解器）+ MuJoCo（物理仿真/可视化）

## 核心架构

### WHAM 推理管线

```text
视频帧 → YOLO 检测 (detector.py) → ViT 特征提取 (extractor.py)
    → MotionEncoder → MotionDecoder → SMPL-X 参数 (pose, shape, trans)
    → [可选] DPVO SLAM (slam.py) → 全局坐标运动
```

- `lib/models/wham.py`：核心网络 `Network`，包含 MotionEncoder、MotionDecoder、TrajectoryDecoder、Integrator
- `lib/models/preproc/detector.py`：YOLO 人体检测
- `lib/models/preproc/extractor.py`：ViT 特征提取
- `lib/models/preproc/slam.py`：DPVO 全局 SLAM（需要正确编译 DPVO 并下载 `checkpoints/dpvo.pth`）
- `lib/models/smpl.py`：SMPL 人体模型包装
- `lib/models/smplify/`：时序 SMPLify 后处理优化

### GMR 重定向管线

```text
SMPL-X 人体运动 → IK 求解器 (mink) → 机器人关节角度 → MuJoCo 可视化
```

- `general_motion_retargeting/motion_retarget.py`：`GeneralMotionRetargeting` 主类，IK 重定向
- `general_motion_retargeting/kinematics_model.py`：`KinematicsModel`，机器人运动学计算
- `general_motion_retargeting/robot_motion_viewer.py`：MuJoCo 可视化（支持异步线程渲染）
- `general_motion_retargeting/params.py`：机器人定义和 IK 配置映射

### 集成流式管线

`handle_wham_gmr.py` 是集成核心，在同一进程中同时运行 WHAM 和 GMR：

1. WHAM 线程：读取视频/摄像头 → 检测/推理 → 生成 SMPL-X → 写入 `.npz` 流文件
2. GMR 线程：轮询 `.npz` 文件 → IK 重定向 → MuJoCo 渲染
3. `run.sh` 通过环境变量控制所有参数

## 常用命令

### 主入口：集成管线

```bash
# 从视频文件运行
OUTPUT_ROOT=output/my_run ROBOT=unitree_g1 RECORD_GMRVIDEO=1 bash run.sh

# 从摄像头实时运行
VIDEO=0 ROBOT=unitree_g1 RECORD_GMRVIDEO=1 bash run.sh

# 高性能模式（跳帧 + 半精度 + 缩小输入）
VIDEO=0 WHAM_DETECT_INTERVAL=2 WHAM_INFER_INTERVAL=2 WHAM_STREAM_SEQ_LEN=12 WHAM_INPUT_SCALE=0.5 WHAM_USE_AMP=1 GMR_TORCH_DEVICE=cuda bash run.sh
```

关键环境变量（完整列表见 [README.md](README.md)）：

- `VIDEO`：视频路径，`0` = 摄像头
- `ROBOT`：目标机器人名（如 `unitree_g1`、`booster_t1`）
- `RECORD_GMRVIDEO`/`RECORD_WHAMVIDEO`：是否录制视频
- `WHAM_USE_AMP`：半精度推理（仅 CUDA）
- `WHAM_DETECT_INTERVAL`/`WHAM_INFER_INTERVAL`：检测/推理跳帧间隔
- `GMR_TORCH_DEVICE`：GMR 使用的 torch 设备（`cpu`/`cuda`）

### WHAM 离线推理

```bash
python demo.py --video <path> --output_dir <dir>
python wham_api.py --video <path> --output_dir <dir>
```

### GMR 离线重定向

```bash
# SMPL-X → 机器人
python scripts/smplx_to_robot.py --smplx_file <path> --robot <robot_name> --save_path <output.pkl>

# BVH → 机器人
python scripts/bvh_to_robot.py --bvh_file <path> --robot <robot_name> --save_path <output.pkl>

# 可视化
python scripts/vis_robot_motion.py --robot <robot_name> --robot_motion_path <path.pkl>
```

### WHAM 训练

```bash
python train.py --config configs/yamls/stage1.yaml
```

## 目录结构

```text
WHAM/
├── run.sh                          # ★ 主入口脚本（集成管线）
├── handle_wham_gmr.py              # ★ WHAM+GMR 集成的生产者-消费者管线
├── demo_stream_mt.py               # WHAM 多线程流式 demo
├── demo.py                         # WHAM 离线推理 demo（含 SLAM）
├── wham_api.py                     # WHAM API 封装
├── train.py                        # WHAM 训练脚本
├── configs/
│   ├── config.py                   # YACS 配置定义
│   ├── constants.py                # 路径/模型常量
│   └── yamls/                      # 训练/推理 YAML 配置
├── lib/
│   ├── models/
│   │   ├── wham.py                 # ★ WHAM 核心网络
│   │   ├── smpl.py                 # SMPL 模型
│   │   ├── preproc/                # 预处理（检测、特征提取、SLAM）
│   │   ├── smplify/                # 时序 SMPLify 优化
│   │   └── layers/                 # 网络层（编码器/解码器等）
│   ├── data/                       # 数据集和数据加载
│   ├── utils/                      # 工具函数（变换、关键点）
│   └── vis/                        # WHAM 可视化渲染
├── general_motion_retargeting/     # ★ GMR 核心库
│   ├── motion_retarget.py          # IK 重定向主类
│   ├── kinematics_model.py         # 机器人运动学
│   ├── robot_motion_viewer.py      # MuJoCo 可视化
│   ├── params.py                   # 机器人定义与 IK 配置
│   ├── ik_configs/                 # IK 映射 JSON 配置
│   └── utils/                      # SMPL/BVH 数据加载
├── scripts/                        # 入口脚本
│   ├── stream_demo_common.py       # 流式公用函数
│   ├── smplx_to_robot_stream.py    # 在线 SMPL-X→机器人后处理
│   └── ...
├── third-party/
│   ├── ViTPose/                    # 人体姿态估计
│   └── DPVO/                       # 视觉 SLAM（可选）
├── assets/
│   ├── body_models/smplx/          # SMPL-X 模型文件
│   └── <robot_name>/               # 各机器人 MuJoCo XML
├── checkpoints/                    # 模型权重
└── docker/                         # Docker 配置
```

## 支持的机器人

核心模型（`assets/` 目录）：

- **Unitree G1** (29 DOF)、**G1 with Hands** (43 DOF)
- **Booster T1**（全身）、**Booster K1** (22 DOF)
- **Stanford ToddlerBot**（科研平台）
- **Fourier N1**（商用）
- **ENGINEAI PM01**（工业）
- **Kuavo S45** (28 DOF)
- **HighTorque Hi** (25 DOF)
- **Galaxea R1 Pro** (24 DOF 轮式)

## 数据流格式

人体运动每帧格式：`dict` of `(body_name, [translation(3), rotation(3)])`
机器人输出：`tuple` of `(base_translation, base_rotation, joint_positions)`

IK 配置文件在 `general_motion_retargeting/ik_configs/`，定义人体部位到机器人关节的映射。

## 关键技术细节

### 四元数约定与 scipy 兼容性（重要）

- **wxyz vs xyzw**：MuJoCo、mink IK 求解器和 IK JSON 配置文件使用 **wxyz**（标量在前）约定；scipy `R.from_quat()` 默认接收 **xyzw**（标量在后）
- **scipy 版本问题**：wham_gmr 环境中的 scipy 1.11.4 **不支持** `scalar_first=True` 关键字参数（该参数在 scipy 1.12+ 才引入）。所有代码必须使用手动索引转换：
  - wxyz → xyzw：`q[[1, 2, 3, 0]]`
  - xyzw → wxyz：`q[[3, 0, 1, 2]]`
- 涉及文件：`motion_retarget.py`、`robot_motion_viewer.py`、`rot_utils.py`、`smpl.py`、`neck_retarget.py`、`xrobot_utils.py`、`xsens_vendor/*.py`

### SMPL vs SMPL-X 模型差异

- WHAM 输出 SMPL 参数（`body_pose` 为 69 维 = 23 关节 × 3）
- 原始 GMR 管线使用 SMPL-X 模型（`body_pose` 为 63 维 = 21 关节 × 3），两者前 21 个关节共享
- `handle_wham_gmr.py` 中的 `tail_params_to_smplx_frame` 截取前 63 维 body_pose，通过 `smplx.create()` 重建 SMPL-X 关节，匹配原始 `scripts/smplx_to_robot_stream.py` 的行为
- IK 配置中的 `rot_offsets` 基于 SMPL-X 输出校准，直接使用 WHAM 的 SMPL 输出会导致关节扭曲

### 坐标系统

- WHAM 输出 **Y-up**（OpenGL 约定），GMR/MuJoCo 使用 **Z-up**
- `handle_wham_gmr.py` 通过 `--coord_fix yup_to_zup` 处理转换：对 global_orient 和 transl 应用 `R_x(90°)` 旋转

### GLFW 窗口兼容性（Linux/Wayland）

- MuJoCo 内部调用 `glfw.default_window_hints()` 会重置自定义窗口提示
- 解决方案（`robot_motion_viewer.py`）：在 `mjv.launch_passive()` 前 monkey-patch `glfw.default_window_hints`，设置 `SCALE_TO_MONITOR=FALSE`、`MAXIMIZED=FALSE` 实现窗口化模式
- 设置 `GDK_SCALE=1` 等环境变量防止 HiDPI 自动缩放

### 其他

- **DPVO 编译**：CUDA 11.x 的 nvcc 最高支持 GCC 10，系统 GCC 11 需安装 `gcc-10 g++-10` 并使用 `CUDAHOSTCXX=/usr/bin/g++-10 pip install --no-build-isolation .`
- **IK 求解器**：mink 库，求解器类型 `daqp`，阻尼默认 5e-1；位置权重 (100) 远大于方向权重 (10)
- **时序平滑**：`SMOOTH_ALPHA`（默认 0.35）控制关节角度平滑
- **人体身高缩放**：基于 `actual_human_height` 自动缩放
- **SLAM 可选性**：未安装 DPVO 时，WHAM 仅估计局部坐标运动
- **流式模式**：`handle_wham_gmr.py` 使用生产者-消费者模式——WHAM 线程持续写入 `.npz` 分块，GMR 线程轮询消费
- **渲染线程**：GMR 可视化在独立线程中运行，避免阻塞主推理循环
- **热启动**：`run.sh` 默认开启一次性 CUDA/PyTorch 预热（`E2E_WARMUP=1`），标记缓存于 `.cache/wham-gmr/`
