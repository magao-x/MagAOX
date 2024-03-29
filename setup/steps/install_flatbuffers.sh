#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
FLATBUFFERS_VERSION="23.5.26"
#
# Flatbuffers
#
cd /opt/MagAOX/vendor
FLATBUFFERS_DIR="flatbuffers-$FLATBUFFERS_VERSION"
if [[ ! -d $FLATBUFFERS_DIR ]]; then
    _cached_fetch https://github.com/google/flatbuffers/archive/v$FLATBUFFERS_VERSION.tar.gz $FLATBUFFERS_DIR.tar.gz
    tar xzf $FLATBUFFERS_DIR.tar.gz
fi
cd $FLATBUFFERS_DIR
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make -j
sudo make install
