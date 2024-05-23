#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -exuo pipefail
sudo mkdir -p /opt/MagAOX/vendor/cuda_rpm
sudo chown :magaox-dev /opt/MagAOX/vendor/cuda_rpm
sudo chmod g+ws /opt/MagAOX/vendor/cuda_rpm
cd /opt/MagAOX/vendor/cuda_rpm
rpmFile=cuda-repo-rhel9-12-4-local-12.4.1_550.54.15-1.x86_64.rpm
_cached_fetch https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/$rpmFile $rpmFile
if ! rpm -q cuda-repo-rhel9-12-4-local-12.4.1_550.54.15-1.x86_64; then
    sudo rpm -i $rpmFile
fi
sudo dnf clean all
sudo dnf -y install cuda-toolkit-12-4
sudo dnf -y module install nvidia-driver:open-dkms