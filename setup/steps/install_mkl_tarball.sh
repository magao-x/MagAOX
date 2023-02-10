#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/vendor
log_info "Began MKL offline package install"
INTEL_MKL_DOWNLOAD=l_onemkl_p_2023.0.0.25398_offline.sh
INTEL_MKL_URL=https://registrationcenter-download.intel.com/akdlm/irc_nas/19138/${INTEL_MKL_DOWNLOAD}
if [[ ! -e $INTEL_MKL_DOWNLOAD ]]; then
    _cached_fetch $INTEL_MKL_URL $INTEL_MKL_DOWNLOAD
fi
if [[ ! -d /opt/intel ]]; then
    sudo sh $INTEL_MKL_DOWNLOAD -a --eula accept --silent
else
    log_warn "/opt/intel already exists"
fi
echo "/opt/intel/oneapi/mkl/latest/lib/intel64" | sudo tee /etc/ld.so.conf.d/mkl.conf || exit 1
sudo ldconfig
log_info "Finished MKL tarball install"