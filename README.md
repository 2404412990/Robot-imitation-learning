# Robot-imitation-learning

根据 [GMR](docs/gmr.md) 和 [WHAM](docs/wham.md) 配好环境，创建 `gmr` 和 `wham` 两个 conda 虚拟环境。

## 运行

示例命令：

```bash
OUTPUT_ROOT=output/my_run ROBOT=unitree_h1 RECORD_GMRVIDEO=1 RECORD_WHAMVIDEO=1 VIDEO=examples/IMG_9732.mov bash run.sh
```

最小可运行命令（全部默认参数）：

```bash
bash run.sh
```

## 参数解释

- `VIDEO`：默认 `examples/IMG_9732.mov`，指定输入视频路径，为0时打开真实摄像头输入。

- `TIME`：只有 `VIDEO=0` 时有效，默认 `0`。当 `TIME>0` 时，真实摄像头录制 `TIME` 秒后自动停止；当 `TIME=0` 时，在终端输入 `q`、按 `Esc` 键或输入 `esc` 停止录制（终端触发停止（`q`/`Esc`/`esc`）后会优先快速关闭预览窗口并结束流程，队列中尚未消费的少量帧可能被丢弃（这是预期行为，用于避免窗口卡死））。

- `ROBOT`：默认 `unitree_g1`，指定重定向机器人类型。

- `ROBOT_PATH`：默认空。可传机器人名（如 `unitree_h1`）或 xml 路径（如 `assets/unitree_g1/g1_mocap_29dof.xml`）。

- `RECORD_VIDEO`：默认 `1`。

- `RECORD_WHAMVIDEO`：默认跟随 `RECORD_VIDEO`，因此默认 `1`；当为 `1` 时会在 `OUTPUT_ROOT/stream_demo` 下输出 WHAM 可视化结果。

- `RECORD_GMRVIDEO`：默认跟随 `RECORD_VIDEO`，因此默认 `1`，当为1时会在屏幕上渲染mujoco窗口。

- `USE_XVFB_GMR`：默认 `0`。当设为 `1` 时，GMR 在虚拟显示中渲染，不会在物理屏幕弹窗。

- `OUTPUT_ROOT`：默认空。为空时默认输出路径为：

- `output/stream_demo`
- `pkl_outputs/my_motion.pkl`
- `pkl_outputs/csv/live_motion.csv`
- `videos/live_stream_robot.mp4`

- 相机默认值：`CAMERA_FOLLOW=1`、`CAMERA_LOOKAT_HEIGHT_OFFSET=0.75`、`CAMERA_ELEVATION=-5.0`、`CAMERA_DISTANCE_SCALE=1.0`、`CAMERA_AZIMUTH` 为空。

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
docker compose -f docker/compose.yml build
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
- 项目目录通过 volume 挂载到容器 `/workspace`，因此你本机当前代码、模型和输出路径都可直接复用。
