#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
source /etc/os-release
if [[ ! $ID == "ubuntu" ]]; then
    log_error "Only installing from package manager on Ubuntu"
    exit 1
fi
sudo apt install -y intel-oneapi-mkl-devel-2022.2.0
echo "source /opt/intel/oneapi/mkl/latest/env/vars.sh intel64" | sudo tee /etc/profile.d/mklvars.sh &>/dev/null || exit 1