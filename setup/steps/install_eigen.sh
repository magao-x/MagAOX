#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
EIGEN_VERSION="3.3.4"
#
# Eigen
#
if [[ ! -e /usr/local/include/Eigen ]]; then
    _cached_fetch http://bitbucket.org/eigen/eigen/get/$EIGEN_VERSION.tar.gz eigen-$EIGEN_VERSION.tar.gz
    tar xzf eigen-$EIGEN_VERSION.tar.gz
    EIGEN_DIR=$(realpath $(find . -type d -name "eigen-eigen-*" | head -n 1))
    ln -sv "$EIGEN_DIR/Eigen" "/usr/local/include/Eigen"
    echo "/usr/local/include/Eigen is now a symlink to $EIGEN_DIR"
fi
