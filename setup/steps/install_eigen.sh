#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
EIGEN_VERSION="3.3.4"
#
# Eigen
#
cd /opt/MagAOX/vendor || exit 1
if [[ ! -e /usr/local/share/pkgconfig/eigen3.pc ]]; then
    EIGEN_DIR="/opt/MagAOX/vendor/eigen-$EIGEN_VERSION"
    if [[ ! -d $EIGEN_DIR ]]; then
        eigenArchive=eigen-$EIGEN_VERSION.tar.gz
        _cached_fetch https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz $eigenArchive || exit 1
        tar xzf $eigenArchive || exit 1
        if [[ ! -d $EIGEN_DIR ]]; then
            mv eigen-*/ $EIGEN_DIR/ || exit 1
        fi
    fi
    if [[ ! -d $EIGEN_DIR/_build ]]; then
        mkdir -p $EIGEN_DIR/_build || exit 1
        cd $EIGEN_DIR/_build || exit 1
        cmake .. || exit 1
    else
        cd $EIGEN_DIR/_build || exit 1
    fi
    sudo make install || exit 1
fi
