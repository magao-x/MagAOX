#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
EIGEN_VERSION="3.3.4"
#
# Eigen
#
if [[ ! -e /usr/local/include/Eigen ]]; then
    EIGEN_DIR="/opt/MagAOX/vendor/eigen-$EIGEN_VERSION"
    if [[ ! -d $EIGEN_DIR ]]; then
        _cached_fetch https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz eigen-$EIGEN_VERSION.tar.gz
        tar xzf eigen-$EIGEN_VERSION.tar.gz
    fi
    ln -sv "$EIGEN_DIR/Eigen" "/usr/local/include/Eigen"
    echo "/usr/local/include/Eigen is now a symlink to $EIGEN_DIR"
fi
