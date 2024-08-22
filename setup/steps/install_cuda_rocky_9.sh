#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
sudo mkdir -p /opt/MagAOX/vendor/cuda_rpm || exit 1
sudo chown :magaox-dev /opt/MagAOX/vendor/cuda_rpm || exit 1
sudo chmod g+ws /opt/MagAOX/vendor/cuda_rpm || exit 1
cd /opt/MagAOX/vendor/cuda_rpm || exit 1
rpmFile=cuda-repo-rhel9-12-4-local-12.4.1_550.54.15-1.x86_64.rpm
_cached_fetch https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/$rpmFile $rpmFile || exit 1
if ! rpm -q cuda-repo-rhel9-12-4-local-12.4.1_550.54.15-1.x86_64; then
    sudo rpm -i $rpmFile || exit 1
fi
sudo dnf clean all || exit 1
sudo dnf -y install cuda-toolkit-12-4 || exit 1
sudo dnf -y module install nvidia-driver:open-dkms || exit 1