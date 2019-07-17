#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
FFTW_VERSION="3.3.8"
#
# FFTW (note: need 3.3.8 or newer, so can't use yum)
#
if [[ ! -d "./fftw-$FFTW_VERSION" ]]; then
    if [[ ! -e fftw-$FFTW_VERSION.tar.gz ]]; then
        curl -OL http://fftw.org/fftw-$FFTW_VERSION.tar.gz
    fi
    tar xzf fftw-$FFTW_VERSION.tar.gz
fi
cd fftw-$FFTW_VERSION
# Following Jared's comprehensive build script: https://gist.github.com/jaredmales/0aacc00b0ce493cd63d3c5c75ccc6cdd
if [ ! -e /usr/local/lib/libfftw3f.a ]; then
    ./configure --enable-float --with-combined-threads --enable-threads --enable-shared
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3.a ]; then
    ./configure --with-combined-threads --enable-threads --enable-shared
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3l.a ]; then
    ./configure --enable-long-double --with-combined-threads --enable-threads --enable-shared
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3q.a ]; then
    ./configure --enable-quad-precision --with-combined-threads --enable-threads --enable-shared
    make
    make install
fi
