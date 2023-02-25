#!/bin/bash
# If not started as root, sudo yourself
export BUILDING_KERNEL_STUFF=1  # disable loading devtoolset-7 for agreement w/ kernel gcc
if [[ "$EUID" != 0 ]]; then
    echo "Becoming root, disabling loading devtoolset-7 for agreement w/ kernel gcc..."
    /usr/bin/sudo --preserve-env=BUILDING_KERNEL_STUFF bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

log_info "Setting up CUDA for $MAGAOX_ROLE"

if [[ $MAGAOX_ROLE == vm || $MAGAOX_ROLE == ci ]]; then
  TMP_CUDA_DIR=$HOME/tmp
  mkdir -p $TMP_CUDA_DIR
  CUDA_FLAGS="--silent --toolkit --tmpdir=$TMP_CUDA_DIR"
elif [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TIC ]]; then
  CUDA_FLAGS="--silent --driver --toolkit --samples"
  export IGNORE_PREEMPT_RT_PRESENCE=1
else
  CUDA_FLAGS="--driver --toolkit --samples"
fi
if [[ $CUDA_FLAGS == *driver* ]]; then
  systemGcc=$(/usr/bin/gcc --version | head -n 1)
  currentGcc=$(gcc --version | head -n 1)

  if [[ $currentGcc != $systemGcc ]]; then
    log_error "You need to use the system GCC ($systemGcc) to build kernel drivers but gcc is $currentGcc"
    exit 1
  fi
fi
CUDA_PACKAGE_DIR=/opt/MagAOX/vendor/cuda
mkdir -p $CUDA_PACKAGE_DIR
cd $CUDA_PACKAGE_DIR
# We use the local CUDA installer (2.5 GB download) to ensure
# we can reinstall without a high-bandwidth connection in a pinch
CUDA_VERSION=11.8
CUDA_RUNFILE=cuda_11.8.0_520.61.05_linux.run
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/$CUDA_RUNFILE
_cached_fetch $CUDA_URL $CUDA_RUNFILE
if [[ ! -e /usr/local/cuda-$CUDA_VERSION ]]; then
    log_info "Starting installation: $CUDA_PACKAGE_DIR/$CUDA_RUNFILE $CUDA_FLAGS"
    bash $CUDA_RUNFILE $CUDA_FLAGS || exit 1
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
  cp /usr/share/doc/NVIDIA_GLX-1.0/samples/nvidia-persistenced-init.tar.bz2 .
  tar xjf nvidia-persistenced-init.tar.bz2
  cd nvidia-persistenced-init
  # NVIDIA's install script adds the user, adds a systemd unit
  # enables it, and starts it.
  sudo bash install.sh systemd
  rm -r $workdir
fi

# Pick up new /etc/ld.so.conf.d/cuda*.conf file
ldconfig
