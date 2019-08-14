#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
TARGET_ENV=$1
log_info "Setting up CUDA for $TARGET_ENV"

if [[ "$TARGET_ENV" == "vm" || "$TARGET_ENV" == "ci" ]]; then
  TMP_CUDA_DIR=$HOME/tmp
  mkdir -p $TMP_CUDA_DIR
  CUDA_FLAGS="--silent --toolkit --tmpdir=$TMP_CUDA_DIR"
elif [[ "$TARGET_ENV" == "instrument" ]]; then
  CUDA_FLAGS="--silent --driver --toolkit --samples"
  export IGNORE_PREEMPT_RT_PRESENCE=1
else
  echo "Unknown TARGET_ENV passed as argument 1"
  exit 1
fi
CUDA_PACKAGE_DIR=./cuda
mkdir -p $CUDA_PACKAGE_DIR
cd $CUDA_PACKAGE_DIR
# We use the local CUDA installer (2.5 GB download) to ensure
# we can reinstall without a high-bandwidth connection in a pinch
CUDA_VERSION=10.1
CUDA_RUNFILE=cuda_10.1.168_418.67_linux.run
CUDA_URL=https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/$CUDA_RUNFILE
if [[ ! -e /usr/local/cuda-$CUDA_VERSION ]]; then
    _cached_fetch $CUDA_URL $CUDA_RUNFILE
    bash $CUDA_RUNFILE $CUDA_FLAGS
fi
echo "export CUDADIR=/usr/local/cuda" > /etc/profile.d/cuda.sh
echo "export CUDA_ROOT=/usr/local/cuda" >> /etc/profile.d/cuda.sh
echo "export PATH=\"\$PATH:/usr/local/cuda/bin\"" >> /etc/profile.d/cuda.sh
