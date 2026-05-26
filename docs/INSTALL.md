# Installation

WHAM and GMR integration requires **Python 3.10**. We provide an integrated [anaconda](https://www.anaconda.com/) environment to run both WHAM and GMR seamlessly.

```bash
# 1. Clone the repo
git clone https://github.com/yohanshin/WHAM.git --recursive
cd WHAM/

# 2. Create Conda environment (Unified for WHAM + GMR)
conda create -n wham_gmr python=3.10 -y
conda activate wham_gmr

# 3. Install PyTorch libraries (Compatible with PyTorch 1.11.0, CUDA 11.3)
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y

# 4. Install PyTorch3D (Fixed Py3.10 wheel)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1110/download.html

# 5. Install WHAM dependencies
# Install older setuptools and mmcv with no-build-isolation first to avoid pkg_resources error
pip install setuptools==59.5.0
pip install --no-build-isolation mmcv==1.3.9
pip install -r requirements.txt

# 6. Install ViTPose runtime deps (workaround for chumpy build isolation issue on newer pip)
python -m pip install --no-build-isolation chumpy==0.70 json-tricks

# 7. Install ViTPose
pip install -v -e third-party/ViTPose

# 8. Install DPVO
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
conda install pytorch-scatter=2.0.9 -c rusty1s -y
# CUDA 11.x nvcc only supports GCC ≤10. System GCC 11 will cause C++ header errors.
# setup.py is patched to use g++-10 via -ccbin flag.
sudo apt install gcc-10 g++-10 -y
# Install DPVO (disable build isolation so setup.py can use torch from current env)

python -m pip install --no-build-isolation .
cd ../..

# 9. Install GMR (General Motion Retargeting)
pip install -e .
```

