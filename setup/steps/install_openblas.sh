#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
cd /opt/MagAOX/vendor || exit 1
log_info "Install OpenBLAS from source"
VERSION=0.3.24
DOWNLOAD_FILE=OpenBLAS-${VERSION}.tar.gz
DOWNLOAD_URL=https://github.com/xianyi/OpenBLAS/releases/download/v${VERSION}/${DOWNLOAD_FILE}
if [[ ! -e $DOWNLOAD_FILE ]]; then
    _cached_fetch $DOWNLOAD_URL $DOWNLOAD_FILE || exit 1
fi
if [[ ! -d ./OpenBLAS-${VERSION} ]]; then
    tar xf $DOWNLOAD_FILE || exit 1
fi
cd ./OpenBLAS-${VERSION} || exit 1
make clean || exit 1
make USE_OPENMP=1 || exit 1
sudo make install PREFIX=/usr/local || exit 1
log_info "Finished OpenBLAS source install"