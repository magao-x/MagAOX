#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
XPA_URL="https://github.com/ericmandel/xpa/archive/v2.1.19.tar.gz"
XPA_VERSION="2.1.19"
XPA_DIR="xpa-$XPA_VERSION"
XPA_TARBALL="$XPA_DIR.tar.gz"

if [[ ! -e $XPA_DIR ]]; then
    if [[ ! -e $XPA_TARBALL ]]; then
        curl -L $XPA_URL > $XPA_TARBALL
    fi
    log_info "Extracting $XPA_TARBALL..."
    tar xzf $XPA_TARBALL
fi
cd $XPA_DIR
./configure
make
make install
