#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
EIGEN_VERSION="3.3.4"
#
# Eigen
#
if [[ ! -e $(readlink "/usr/local/include/Eigen") ]]; then
    if [[ ! -e eigen-$EIGEN_VERSION.tar.gz ]]; then
        curl -L http://bitbucket.org/eigen/eigen/get/$EIGEN_VERSION.tar.gz > eigen-$EIGEN_VERSION.tar.gz
    fi
    tar xzf eigen-$EIGEN_VERSION.tar.gz
    EIGEN_DIR=$(realpath $(find . -type d -name "eigen-eigen-*" | head -n 1))
    ln -sv "$EIGEN_DIR/Eigen" "/usr/local/include/Eigen"
    echo "/usr/local/include/Eigen is now a symlink to $EIGEN_DIR"
fi
