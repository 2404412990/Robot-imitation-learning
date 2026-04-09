#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Please run as a normal user, not root."
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required but not found."
  exit 1
fi

disable_stale_cuda_sources() {
  echo "[pre] Disable stale/invalid CUDA apt sources"

  # Disable old third-party repo files first. Fresh ones will be regenerated later.
  for f in /etc/apt/sources.list.d/docker.list /etc/apt/sources.list.d/nvidia-container-toolkit.list; do
    [[ -f "${f}" ]] || continue
    sudo mv "${f}" "${f}.disabled"
    echo "  - disabled stale third-party source ${f}"
  done

  # Some machines have typo files like cuda.listc that trigger apt warnings.
  for f in /etc/apt/sources.list.d/*.listc; do
    [[ -e "${f}" ]] || continue
    sudo mv "${f}" "${f}.disabled"
    echo "  - disabled ${f}"
  done

  # Disable dead local CUDA repo and stale NVIDIA CUDA repo entries that often break apt update.
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
    [[ -f "${f}" ]] || continue
    sudo sed -i -E \
        -e '/file:\/+var\/cuda-repo/ s|^|# disabled stale local CUDA repo: |' \
      -e '/developer\.download\.nvidia\.(cn|com)\/compute\/cuda\/repos/ s|^|# disabled stale NVIDIA CUDA repo: |' \
      "${f}"
  done

  # Deb822 format source files (Ubuntu 22.04+ may use .sources). Disable any stale CUDA entries.
  for f in /etc/apt/sources.list.d/*.sources; do
    [[ -f "${f}" ]] || continue
    if sudo grep -Eq 'file:/+var/cuda-repo|developer\.download\.nvidia\.(cn|com)/compute/cuda/repos' "${f}"; then
      sudo mv "${f}" "${f}.disabled"
      echo "  - disabled ${f} (deb822 stale CUDA source)"
    fi
  done
}

pick_first_reachable() {
  local test_path="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if curl -fsSLI --connect-timeout 5 --max-time 10 "${candidate}${test_path}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

disable_stale_cuda_sources

echo "[1/4] Install Docker CE + Compose plugin"
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

. /etc/os-release

sudo install -m 0755 -d /etc/apt/keyrings

if [[ -n "${DOCKER_BASE:-}" ]]; then
  echo "Using Docker apt repo from env: ${DOCKER_BASE}"
else
  DOCKER_BASE="$(pick_first_reachable "/dists/${VERSION_CODENAME}/Release" \
    "https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu" \
    "https://mirrors.aliyun.com/docker-ce/linux/ubuntu" \
    "https://download.docker.com/linux/ubuntu")" || {
    echo "ERROR: Could not reach Docker apt repo (official or mirrors)."
    echo "Please check DNS/network/proxy settings, then retry."
    exit 1
  }
fi
echo "Using Docker apt repo: ${DOCKER_BASE}"

curl -fsSL "${DOCKER_BASE}/gpg" | sudo gpg --dearmor --batch --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] ${DOCKER_BASE} ${VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

if id -nG "${USER}" | grep -qw docker; then
  echo "User ${USER} is already in docker group."
else
  sudo usermod -aG docker "${USER}"
  echo "Added ${USER} to docker group."
fi

echo "[2/4] Install NVIDIA Container Toolkit"
distribution=$(. /etc/os-release;echo ${ID}${VERSION_ID})

if [[ -n "${NVIDIA_BASE:-}" ]]; then
  echo "Using NVIDIA toolkit repo from env: ${NVIDIA_BASE}"
else
  NVIDIA_BASE="$(pick_first_reachable "/${distribution}/libnvidia-container.list" \
    "https://mirrors.tuna.tsinghua.edu.cn/libnvidia-container" \
    "https://nvidia.github.io/libnvidia-container")" || {
    echo "ERROR: Could not reach NVIDIA container toolkit repo."
    echo "Please check DNS/network/proxy settings, or set up accessible mirror."
    exit 1
  }
fi
echo "Using NVIDIA toolkit repo: ${NVIDIA_BASE}"

curl -fsSL "${NVIDIA_BASE}/gpgkey" \
  | sudo gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L "${NVIDIA_BASE}/${distribution}/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "[3/4] Basic checks"
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

echo "[4/4] Done"
echo "Please log out and log in again (or run: newgrp docker) before using docker without sudo."
