# Windows Installation

WHAM + GMR 集成系统在 Windows 上的环境安装指南。要求 **Python 3.10**，使用 [Anaconda](https://www.anaconda.com/) 管理环境。

## 前置条件

| 软件 | 用途 | 下载 |
|------|------|------|
| **Anaconda / Miniconda** | Python 环境管理 | [anaconda.com](https://www.anaconda.com/download) |
| **CUDA Toolkit 11.3** | DPVO/PyTorch3D CUDA 编译 (nvcc) | [NVIDIA CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) |
| **Visual Studio 2019 Community** | C++ 编译器 (cl.exe)，**必须用 2019** | [VS 2019](https://visualstudio.microsoft.com/vs/older-downloads/) |

> **重要**：即使系统已安装 CUDA 13.x，仍需额外安装 CUDA 11.3 Toolkit（多版本共存没问题）。
> Visual Studio **必须用 2019**，VS 2022 会导致 PyTorch3D 及 DPVO 编译失败。

### VS 2019 安装选项
安装时勾选 **"使用 C++ 的桌面开发"** 工作负载。

### 环境变量设置
安装 CUDA 11.3 后，确保编译时使用正确的 CUDA 版本（而非系统 CUDA 13.x）：

```powershell
# 在编译 PyTorch3D / DPVO 之前执行
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
```

验证：

```powershell
conda --version
nvcc --version    # 应显示 CUDA 11.3，而非 13.x
```

## 安装步骤

以下命令在 **"x64 Native Tools Command Prompt for VS 2019"** 中执行（从开始菜单找到该快捷方式，打开后先 `conda activate wham_gmr`）：

> 也可以使用 Anaconda PowerShell Prompt，但需先运行 VS 2019 的环境初始化脚本：
> ```powershell
> & "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
> ```

### 1. 克隆仓库

```bash
git clone https://github.com/yohanshin/WHAM.git --recursive
cd WHAM/
```

### 2. 创建 Conda 环境

```bash
conda create -n wham_gmr python=3.10 -y
conda activate wham_gmr
```

### 3. 安装 PyTorch (CUDA 11.3)

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
```

验证：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 4. 安装 PyTorch3D (从源码编译)

conda-forge 没有 Windows 预编译包，必须从源码编译。

```bash
# 4a. 安装依赖
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y

# 4b. 下载 NVIDIA CUB 1.11.0 (CUDA 11.3 需要)
# 浏览器打开 https://github.com/NVIDIA/cub/releases/tag/1.11.0
# 下载 cub-1.11.0.zip，解压到任意目录，然后：
$env:CUB_HOME = "C:\path\to\cub-1.11.0"

# 4c. 确保使用 CUDA 11.3
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

# 4d. 克隆并编译
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout v0.7.4

$env:DISTUTILS_USE_SDK = "1"
$env:PYTORCH3D_NO_NINJA = "1"

# 修复 setup.py 中 MSVC 不识别的 -std=c++17 标志
(Get-Content setup.py) -replace '-std=c\+\+17', '' | Set-Content setup.py
set KMP_DUPLICATE_LIB_OK=TRUE
python setup.py install
cd ..
```

**如果编译失败**：PyTorch3D 仅用于 WHAM 的可视化渲染窗口（`RECORD_WHAMVIDEO`），不影响 GMR 机器人重定向和 MuJoCo 可视化。可以安全跳过步骤 4。

### 5. 安装 WHAM 依赖

```bash
pip install setuptools==59.5.0
pip install --no-build-isolation mmcv==1.3.9
pip install -r requirements.txt
```

### 6. 安装 ViTPose

```bash
python -m pip install --no-build-isolation chumpy==0.70 json-tricks
pip install -v -e third-party/ViTPose
```

### 7. 安装 DPVO (可选 — 全局 SLAM)

前置条件：
- CUDA 11.3 Toolkit nvcc 在 PATH 中
- Eigen 3.4.0 已下载
- VS 2019 环境已配置

```bash
# 7a. 下载 Eigen 3.4.0
# 使用 GitHub 镜像（GitLab 国内可能无法访问），浏览器：https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
# 解压到 third-party/DPVO/thirdparty/ 目录下，确保路径为：
#   third-party/DPVO/thirdparty/eigen-3.4.0/Eigen/...

# 或使用 PowerShell 下载：
Invoke-WebRequest -Uri "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" -OutFile "$env:TEMP\eigen-3.4.0.zip"
Expand-Archive -Path "$env:TEMP\eigen-3.4.0.zip" -DestinationPath "third-party/DPVO/thirdparty/"


# 7b. 安装 pytorch-scatter
conda install pytorch-scatter=2.0.9 -c rusty1s -y

# 7c. 确保使用 CUDA 11.3（非系统 CUDA 13.x）
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

# 验证 nvcc 版本
nvcc --version  # 必须显示 CUDA 11.3

# 7d. 运行 Windows 兼容性补丁
cd third-party/DPVO
powershell -ExecutionPolicy Bypass -File patch_windows.ps1
cd ../..

# 7e. 安装
cd third-party/DPVO
activate base
conda activate wham_gmr
set CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
set  KMP_DUPLICATE_LIB_OK=TRUE
set DISTUTILS_USE_SDK=1
python -m pip install --no-build-isolation .
cd ../..
```

如果 DPVO 编译仍然失败，可以跳过——未安装 DPVO 仅缺少全局 SLAM 相机运动估计，WHAM 仍可在局部坐标中完成人体运动推理。

### 8. 安装 GMR

```bash
pip install -e .
```

> 如果遇到 `UnicodeDecodeError: 'gbk' codec can't decode byte ...`，请确认 `setup.py` 中的 `open("README.md")` 已改为 `open("README.md", encoding="utf-8")`。

## 常见问题

### CUDA 版本不匹配：`The detected CUDA version (13.2) mismatches ... PyTorch (11.3)`

系统安装了多个 CUDA 版本，编译时需要指向 11.3：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
nvcc --version  # 确认输出为 CUDA 11.3
```

### `cl.exe` 找不到 / `[WinError 2] 系统找不到指定的文件`

VS 2019 的 C++ 编译器未在 PATH 中。从开始菜单打开 **"x64 Native Tools Command Prompt for VS 2019"** 后再执行编译命令。

### mmcv 编译失败

尝试使用 openmim 安装预编译版本：

```bash
pip install openmim
mim install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
```

### PyTorch3D 编译错误 `ignoring unknown option '-std=c++17'`

```powershell
(Get-Content setup.py) -replace '-std=c\+\+17', '' | Set-Content setup.py
```

### `pip install -e .` 报 UnicodeDecodeError

setup.py 读取 README.md 时编码问题。确保 `setup.py` 中 `open("README.md")` 已改为 `open("README.md", encoding="utf-8")`（最新代码已修复）。

## 环境验证

安装完成后，验证核心依赖：

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import smplx; print('SMPL-X:', smplx.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mujoco; print('MuJoCo OK')"
python -c "import mink; print('mink IK OK')"
python -c "from general_motion_retargeting import GeneralMotionRetargeting; print('GMR OK')"
```

全部输出 OK 即表示环境就绪，可以运行 `.\run.ps1`。
