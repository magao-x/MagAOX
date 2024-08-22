#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
cd /opt/MagAOX/vendor

FFTW_VERSION="3.3.8"
#
# FFTW (note: need 3.3.8 or newer, so can't use yum)
#
if [[ ! -d "./fftw-$FFTW_VERSION" ]]; then
    if [[ ! -e fftw-$FFTW_VERSION.tar.gz ]]; then
        _cached_fetch http://fftw.org/fftw-$FFTW_VERSION.tar.gz fftw-$FFTW_VERSION.tar.gz || exit 1
    fi
    tar xzf fftw-$FFTW_VERSION.tar.gz || exit 1
fi
cd fftw-$FFTW_VERSION
# Following Jared's comprehensive build script: https://gist.github.com/jaredmales/0aacc00b0ce493cd63d3c5c75ccc6cdd
if [ ! -e /usr/local/lib/libfftw3f.a ]; then
    ./configure --enable-float --with-combined-threads --enable-threads --enable-shared || exit 1
    make || exit 1
    make install || exit 1
fi
if [ ! -e /usr/local/lib/libfftw3.a ]; then
    ./configure --with-combined-threads --enable-threads --enable-shared || exit 1
    make || exit 1
    make install || exit 1
fi
if [ ! -e /usr/local/lib/libfftw3l.a ]; then
    ./configure --enable-long-double --with-combined-threads --enable-threads --enable-shared || exit 1
    make || exit 1
    make install || exit 1
fi
if [[ $(uname -p) == "x86_64" ]]; then
    # libquadmath is part of gcc and not available on ARM
    if [ ! -e /usr/local/lib/libfftw3q.a ]; then
        ./configure --enable-quad-precision --with-combined-threads --enable-threads --enable-shared || exit 1
        make || exit 1
        make install || exit 1
    fi
fi