#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
CFITSIO_VERSION="3.47"
#
# CFITSIO
#
if [[ ! -d ./cfitsio-$CFITSIO_VERSION ]]; then
    if [[ ! -e cfitsio-$CFITSIO_VERSION.tar.gz ]]; then
        curl -OL http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-$CFITSIO_VERSION.tar.gz
    fi
    tar xzf cfitsio-$CFITSIO_VERSION.tar.gz
fi
cd cfitsio-$CFITSIO_VERSION
if [[ ! (( -e /usr/local/lib/libcfitsio.a ) && ( -e /usr/local/lib/libcfitsio.so )) ]]; then
    ./configure --prefix=/usr/local
    make
    make shared
    make install
fi
