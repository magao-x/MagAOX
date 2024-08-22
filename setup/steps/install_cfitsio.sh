#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
cd /opt/MagAOX/vendor || exit 1
CFITSIO_VERSION="3.47"
#
# CFITSIO
#
if [[ ! -d ./cfitsio-$CFITSIO_VERSION ]]; then
    _cached_fetch http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-$CFITSIO_VERSION.tar.gz cfitsio-$CFITSIO_VERSION.tar.gz
    tar xzf cfitsio-$CFITSIO_VERSION.tar.gz || exit 1
fi
cd cfitsio-$CFITSIO_VERSION || exit 1
if [[ ! (( -e /usr/local/lib/libcfitsio.a ) && ( -e /usr/local/lib/libcfitsio.so )) ]]; then
    ./configure --prefix=/usr/local || exit 1
    make || exit 1
    make shared || exit 1
    make install || exit 1
fi
