#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
CMAKE_VERSION="3.26.4"

cd /opt/MagAOX/vendor
if [[ ! -d $CMAKE_DIR ]]; then
    _cached_fetch https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh cmake-${CMAKE_VERSION}-linux-x86_64.sh
    tar xzf $CMAKE_DIR.tar.gz
fi
cd $CMAKE_DIR
sudo bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --skip-license || exit_error "Installing CMake failed"