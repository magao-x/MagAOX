#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

log_info "Building flatlogs"
cd flatlogs/src
make
make install

log_info "Building MagAOX"
cd ../..
make setup
if [[ $VM_KIND != none || $MAGAOX_ROLE == container || $MAGAOX_ROLE == TOC ]]; then
    if ! grep 'NEED_CUDA = no' local/common.mk; then
        echo 'NEED_CUDA = no' >> local/common.mk
    fi
fi

make all
make install
