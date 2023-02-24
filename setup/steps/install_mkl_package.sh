#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
source /etc/os-release
if [[ ! $ID == "ubuntu" ]]; then
    log_error "Only installing from package manager on Ubuntu"
    exit 1
fi
set -eo pipefail
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install -y intel-oneapi-mkl-devel-2022.2.0 || error_exit "Couldn't install MKL"
echo "source /opt/intel/oneapi/mkl/latest/env/vars.sh intel64" | sudo tee /etc/profile.d/mklvars.sh &>/dev/null || exit 1
echo "/opt/intel/oneapi/mkl/latest/lib/intel64" | sudo tee /etc/ld.so.conf.d/mkl.conf || exit 1
sudo ldconfig
