#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/vendor
log_info "Install OpenBLAS from source"
VERSION=0.3.24
DOWNLOAD_FILE=OpenBLAS-${VERSION}.tar.gz
DOWNLOAD_URL=https://github.com/xianyi/OpenBLAS/releases/download/v${VERSION}/${DOWNLOAD_FILE}
if [[ ! -e $DOWNLOAD_FILE ]]; then
    _cached_fetch $DOWNLOAD_URL $DOWNLOAD_FILE
fi
if [[ ! -d ./OpenBLAS-${VERSION} ]]; then
    tar xf $DOWNLOAD_FILE
fi
cd ./OpenBLAS-${VERSION}
make clean
make USE_OPENMP=1
sudo make install PREFIX=/usr/local
log_info "Finished OpenBLAS source install"