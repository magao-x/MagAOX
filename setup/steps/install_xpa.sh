#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
XPA_URL="https://github.com/ericmandel/xpa/archive/v2.1.19.tar.gz"
XPA_VERSION="2.1.19"
XPA_DIR="xpa-$XPA_VERSION"
XPA_TARBALL="$XPA_DIR.tar.gz"

if [[ ! -e /usr/local/lib/pkgconfig/xpa.pc ]]; then
    if [[ ! -e $XPA_DIR ]]; then
        _cached_fetch $XPA_URL $XPA_TARBALL || exit 1
        log_info "Extracting $XPA_TARBALL..."
        tar xzf $XPA_TARBALL || exit 1
    fi
    cd $XPA_DIR || exit 1
    ./configure || exit 1
    make || exit 1
    make install || exit 1
fi
