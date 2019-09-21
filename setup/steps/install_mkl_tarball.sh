#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/vendor

INTEL_MKL_VERSION=l_mkl_2019.4.243
INTEL_MKL_TARBALL=$INTEL_MKL_VERSION.tgz
INTEL_MKL_TARBALL_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/15540/$INTEL_MKL_TARBALL
if [[ ! -e l_mkl_2019.4.243 ]]; then
    _cached_fetch $INTEL_MKL_TARBALL_URL $INTEL_MKL_TARBALL
    log_info "Extracting $INTEL_MKL_TARBALL..."
    tar xzf $INTEL_MKL_TARBALL
fi
cd $INTEL_MKL_VERSION
if ! grep ACCEPT_EULA=accept ./silent.cfg; then
    sed -i s/ACCEPT_EULA=decline/ACCEPT_EULA=accept/ silent.cfg
fi
if [[ ! -d /opt/intel ]]; then
    ./install.sh -s silent.cfg
else
    log_warn "/opt/intel already exists. Run sudo $PWD/install.sh interactively to uninstall before reinstalling"
fi
echo "source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64" > /etc/profile.d/mklvars.sh
