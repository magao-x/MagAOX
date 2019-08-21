#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

TARGET_ENV=$1

if [[ "$TARGET_ENV" == "vm" || "$TARGET_ENV" == "workstation" ]]; then
  CMAKE_FLAGS=""
elif [[ "$TARGET_ENV" == "instrument" || "$TARGET_ENV" == "ci" ]]; then
  CMAKE_FLAGS="-DUSE_CUDA=YES -DUSE_MAGMA=YES"
else
  echo "Unknown TARGET_ENV passed as argument 1"
  exit 1
fi

if [[ ! -d ./cacao ]]; then
    git clone --recursive --branch dev https://github.com/magao-x/cacao.git cacao
    cd ./cacao
    git config core.sharedRepository true
    git remote add upstream https://github.com/cacao-org/cacao.git
else
    cd ./cacao
    git pull
fi
CACAO_ABSPATH=$PWD

mkdir -p _build
cd _build
if [[ ! -e Makefile ]]; then
  cmake ../ $CMAKE_FLAGS
fi
make
sudo make install
if [[ ! -e /usr/local/bin/milk ]]; then
    sudo ln -s /usr/local/bin/cacao /usr/local/bin/milk
fi
echo "export PATH=\$PATH:$CACAO_ABSPATH/src/CommandLineInterface/scripts" | sudo tee /etc/profile.d/cacao_scripts.sh

if [[ "$TARGET_ENV" != "ci" ]]; then
  if ! grep -q "/milk/shm" /etc/fstab; then
    echo "tmpfs /milk/shm tmpfs rw,nosuid,nodev" | sudo tee -a /etc/fstab
    sudo mkdir -p /milk/shm
    log_success "Created /milk/shm tmpfs mountpoint"
    sudo mount /milk/shm
    log_success "Mounted /milk/shm"
  else
    log_info "Skipping /milk/shm mount setup"
  fi
fi
