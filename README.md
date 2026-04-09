# Robot-imitation-learning

根据 [GMR](docs/gmr.md) 和 [WHAM](docs/wham.md) 配好环境，创建 `gmr` 和 `wham` 两个 conda 虚拟环境。

`checkpoints`和`dataset`可在[链接](https://pan.baidu.com/s/1fVf2eA1OzdRv70M4gm2wSA?pwd=8pnu) 下载

## 运行

示例命令：

```bash
OUTPUT_ROOT=output/my_run \
ROBOT=unitree_h1 \
RECORD_GMRVIDEO=1 \
RECORD_WHAMVIDEO=1 \
VIDEO=examples/IMG_9732.mov \
bash run.sh
```

或者

```bash
OUTPUT_ROOT=output/faster_run \
RECORD_WHAMVIDEO=1 \
RECORD_GMRVIDEO=1 \
VIDEO=examples/IMG_9732.mov \
WHAM_DETECT_INTERVAL=2 \
WHAM_INFER_INTERVAL=2 \
WHAM_STREAM_SEQ_LEN=12 \
WHAM_INPUT_SCALE=0.5 \
GMR_TORCH_DEVICE=cuda \
bash run.sh
```

## 参数解释

- `VIDEO`：默认 `examples/IMG_9732.mov`，指定输入视频路径，为0时打开真实摄像头输入。

- `TIME`：只有 `VIDEO=0` 时有效，默认 `0`。当 `TIME>0` 时，真实摄像头录制 `TIME` 秒后自动停止；当 `TIME=0` 时，在终端输入 `q`、按 `Esc` 键或输入 `esc` 停止录制（终端触发停止（`q`/`Esc`/`esc`）后会优先快速关闭预览窗口并结束流程，队列中尚未消费的少量帧可能被丢弃（这是预期行为，用于避免窗口卡死））。

- `ROBOT`：默认 `unitree_g1`，指定重定向机器人类型。

- `ROBOT_PATH`：默认空。可传机器人名（如 `unitree_h1`）或 xml 路径（如 `assets/unitree_g1/g1_mocap_29dof.xml`）。

- `RECORD_VIDEO`：默认 `1`。

- `RECORD_WHAMVIDEO`：默认跟随 `RECORD_VIDEO`，因此默认 `1`；当为 `1` 时会在 `OUTPUT_ROOT/stream_demo` 下输出 WHAM 可视化结果，并在有图形环境（`DISPLAY` 可用）时弹出 WHAM 预览窗口（无论输入是文件视频还是摄像头）。

- `RECORD_GMRVIDEO`：默认跟随 `RECORD_VIDEO`，因此默认 `1`，当为1时会在屏幕上渲染mujoco窗口。

- `USE_XVFB_GMR`：默认 `0`。当设为 `1` 时，GMR 在虚拟显示中渲染，不会在物理屏幕弹窗。

- `OUTPUT_ROOT`：默认空。为空时默认输出路径为：

- `output/stream_demo`
- `pkl_outputs/my_motion.pkl`
- `pkl_outputs/csv/live_motion.csv`
- `videos/live_stream_robot.mp4`

- 相机默认值：`CAMERA_FOLLOW=0`、`CAMERA_LOOKAT_HEIGHT_OFFSET=0.45`、`CAMERA_ELEVATION=12.0`、`CAMERA_DISTANCE_SCALE=0.85`、`CAMERA_AZIMUTH` 为空。
- 若希望镜头固定跟随，可在命令前加 `CAMERA_FOLLOW=1`。

- `ROOT_ORIGIN_OFFSET`：默认 `0`。默认保留机器人全局平移，不再将首帧位置重置为原点；如需回到“以起点为原点”的相对轨迹可设为 `1`。

- `WHAM_USE_AMP`：默认 `0`。开启半精度推理（CUDA 下）以提升 WHAM 吞吐。

- `WHAM_DETECT_INTERVAL`：默认 `1`。每 N 帧做一次完整检测，其余帧复用跟踪结果；增大可提速但会牺牲精度。

- `WHAM_INFER_INTERVAL`：默认 `1`。每 N 帧执行一次完整 WHAM 推理；大于 1 时中间帧复用上次结果，显著提速但动作细节会变粗。

- `WHAM_STREAM_SEQ_LEN`：默认 `16`。WHAM 时序窗口长度；减小（如 `8`）可提速但会损失时序稳定性。

- `WHAM_INPUT_SCALE`：默认 `1.0`。输入缩放比例（`0.1~1.0`），越小越快。

- `GMR_TORCH_DEVICE`：默认 `cpu`。控制 GMR 后处理/FK 的 torch 设备（`cpu`/`cuda`/`auto`）。

## docker配置

下面这套配置已对齐当前仓库的 `run.sh`（WHAM + GMR 流式链路）。

### 1. 安装 Docker 与 NVIDIA Container Toolkit（Ubuntu 22.04）

仓库已提供安装脚本：`docker/install_docker_ubuntu.sh`。

```bash
bash docker/install_docker_ubuntu.sh
```

安装完成后，重新登录一次终端（或执行 `newgrp docker`），再检查：

```bash
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. 构建镜像

仓库已提供：

- Dockerfile：`docker/Dockerfile`
- Compose：`docker/compose.yml`
- Docker ignore：`.dockerignore`

在仓库根目录执行：

```bash
docker compose -f docker/compose.yml build --no-cache wham-gmr
```

(时间可能很长)

### 3. 启动容器（带图形界面）

如果需要 MuJoCo 可视化窗口，先放开本机 X11：

```bash
xhost +local:docker
```

启动容器：

```bash
docker compose -f docker/compose.yml run --rm wham-gmr
```

进入容器后，在 `/workspace` 下执行（与本机命令一致）：

```bash
OUTPUT_ROOT=output/my_run ROBOT=unitree_h1 RECORD_GMRVIDEO=1 RECORD_WHAMVIDEO=1 VIDEO=examples/IMG_9732.mov bash run.sh
```

容器内默认已设置：

- `WHAM_PYTHON=/opt/conda/envs/wham/bin/python`
- `GMR_PYTHON=/opt/conda/envs/gmr/bin/python`

收尾时建议恢复 X11 权限：

```bash
xhost -local:docker
```

### 4. 无界面服务器运行

不依赖宿主机窗口，直接让 GMR 在虚拟显示渲染：

```bash
docker compose -f docker/compose.yml run --rm \
  -e USE_XVFB_GMR=1 \
  -e VIDEO=0 \
  -e TIME=10 \
  -e RECORD_GMRVIDEO=1 \
  -e RECORD_WHAMVIDEO=1 \
  -e OUTPUT_ROOT=output/my_run \
  wham-gmr \
  bash run.sh
```

### 5. 打包镜像（导出/导入）

导出：

```bash
docker save wham-gmr:local | gzip > wham-gmr_local.tar.gz
```

导入：

```bash
gzip -dc wham-gmr_local.tar.gz | docker load
```

### 6. 说明

- 当前 Dockerfile 默认优先覆盖 `run.sh` 所需链路；`demo.py` 完整 SLAM 路径若需 DPVO CUDA 编译，可在镜像内按 `third-party/DPVO` 的官方步骤补装。
- 项目目录通过 volume 挂载到容器 `/workspace`
