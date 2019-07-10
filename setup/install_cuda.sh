#!/bin/bash
set -exuo pipefail
CUDA_RPM_DIR=$DEPSROOT/cuda
mkdir -p $CUDA_RPM_DIR
cd $CUDA_RPM_DIR
CUDA_RPM_FILE="cuda-repo-rhel7-10-1-local-10.1.168-418.67-1.0-1.x86_64.rpm"
if [[ ! -e $CUDA_RPM_FILE ]]; then
    curl -OL "https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/$CUDA_RPM_FILE"
fi
rpm -i $CUDA_RPM_FILE || true
yum clean all
yum install -y cuda
