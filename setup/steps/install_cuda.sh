#!/bin/bash
# If not started as root, sudo yourself
if [[ "$EUID" != 0 ]]; then
    sudo bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILDING_KERNEL_STUFF=1  # disable loading devtoolset-7 for agreement w/ kernel gcc
source $DIR/../_common.sh
set -euo pipefail

log_info "Setting up CUDA for $MAGAOX_ROLE"

if [[ $MAGAOX_ROLE == vm || $MAGAOX_ROLE == ci ]]; then
  TMP_CUDA_DIR=$HOME/tmp
  mkdir -p $TMP_CUDA_DIR
  CUDA_FLAGS="--silent --toolkit --tmpdir=$TMP_CUDA_DIR"
elif [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TCC ]]; then
  CUDA_FLAGS="--silent --driver --toolkit --samples"
  export IGNORE_PREEMPT_RT_PRESENCE=1
else
  CUDA_FLAGS="--driver --toolkit --samples"
fi
CUDA_PACKAGE_DIR=/opt/MagAOX/vendor/cuda
mkdir -p $CUDA_PACKAGE_DIR
cd $CUDA_PACKAGE_DIR
# We use the local CUDA installer (2.5 GB download) to ensure
# we can reinstall without a high-bandwidth connection in a pinch
CUDA_VERSION=10.1
CUDA_RUNFILE=cuda_10.1.168_418.67_linux.run
CUDA_URL=https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/$CUDA_RUNFILE
_cached_fetch $CUDA_URL $CUDA_RUNFILE
if [[ ! -e /usr/local/cuda-$CUDA_VERSION ]]; then
    bash $CUDA_RUNFILE $CUDA_FLAGS
else
    log_info "Existing CUDA install found in /usr/local/cuda-$CUDA_VERSION"
    log_info "sudo /usr/local/cuda-$CUDA_VERSION/bin/cuda-uninstaller to uninstall"
fi
echo "export CUDADIR=/usr/local/cuda" > /etc/profile.d/cuda.sh
echo "export CUDA_ROOT=/usr/local/cuda" >> /etc/profile.d/cuda.sh
echo "export PATH=\"\$PATH:/usr/local/cuda/bin\"" >> /etc/profile.d/cuda.sh

# Install nvidia-persistenced
if [[ $MAGAOX_ROLE != vm && $MAGAOX_ROLE != ci && ! -e /usr/lib/systemd/system/nvidia-persistenced.service ]]; then
  workdir=/tmp/persistenced_setup_$(date +%s)
  mkdir $workdir
  cd $workdir
  cp /usr/share/doc/NVIDIA_GLX-1.0/sample/nvidia-persistenced-init.tar.bz2 .
  tar xjf nvidia-persistenced-init.tar.bz2
  cd nvidia-persistenced-init
  # NVIDIA's install script adds the user, adds a systemd unit
  # enables it, and starts it.
  sudo bash install.sh systemd
  rm -r $workdir
fi
