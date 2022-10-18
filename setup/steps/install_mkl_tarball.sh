#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/vendor
log_info "Began MKL offline package install"
log_error "MKL download link broken again..."
exit 1
INTEL_MKL_VERSION=l_onemkl_p_2022.2.0.8748
INTEL_MKL_DOWNLOAD=${INTEL_MKL_VERSION}_offline.sh
INTEL_MKL_URL=https://registrationcenter-download.intel.com/akdlm/irc_nas/18721/$INTEL_MKL_DOWNLOAD
if [[ ! -e $INTEL_MKL_DOWNLOAD ]]; then
    _cached_fetch $INTEL_MKL_URL $INTEL_MKL_DOWNLOAD
fi
if [[ ! -d /opt/intel ]]; then
    sudo sh $INTEL_MKL_DOWNLOAD -a --eula accept --silent
else
    log_warn "/opt/intel already exists"
fi
# echo "source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64" > /etc/profile.d/mklvars.sh
# echo "/opt/intel/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin" | sudo tee /etc/ld.so.conf.d/mkl.conf
ldconfig
log_info "Finished MKL tarball install"
