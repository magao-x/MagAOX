#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
CMAKE_VERSION="3.26.4"
CMAKE_DIR="cmake-${CMAKE_VERSION}"
cd /opt/MagAOX/vendor
if [[ ! -d $CMAKE_DIR ]]; then
    mkdir -p $CMAKE_DIR
    cd $CMAKE_DIR
    _cached_fetch https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh cmake-${CMAKE_VERSION}-linux-x86_64.sh
    sudo bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --skip-license || exit_with_error "Installing CMake failed"
fi
