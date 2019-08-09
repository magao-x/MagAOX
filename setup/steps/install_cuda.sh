#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
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
    sh $CUDA_RUNFILE --silent --drivers --toolkit
fi
echo "export CUDADIR=/usr/local/cuda" > /etc/profile.d/cuda.sh
echo "export CUDA_ROOT=/usr/local/cuda" >> /etc/profile.d/cuda.sh
echo "export PATH=\"\$PATH:/usr/local/cuda/bin\"" >> /etc/profile.d/cuda.sh
