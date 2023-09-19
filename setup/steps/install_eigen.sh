#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
EIGEN_VERSION="3.3.4"
#
# Eigen
#
cd /opt/MagAOX/vendor
if [[ ! -e /usr/local/share/pkgconfig/eigen3.pc ]]; then
    EIGEN_DIR="/opt/MagAOX/vendor/eigen-$EIGEN_VERSION"
    if [[ ! -d $EIGEN_DIR ]]; then
        eigenArchive=eigen-$EIGEN_VERSION.tar.gz
        _cached_fetch https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz $eigenArchive
        tar xzf $eigenArchive
        if [[ ! -d $EIGEN_DIR ]]; then
            mv eigen-*/ $EIGEN_DIR/
        fi
    fi
    mkdir $EIGEN_DIR/_build
    cd $EIGEN_DIR/_build
    cmake ..
    sudo make install
fi
