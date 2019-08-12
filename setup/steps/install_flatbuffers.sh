#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
FLATBUFFERS_VERSION="1.9.0"
#
# Flatbuffers
#
FLATBUFFERS_DIR="./flatbuffers-$FLATBUFFERS_VERSION"
if [[ ! -d $FLATBUFFERS_DIR ]]; then
    _cached_fetch https://github.com/google/flatbuffers/archive/v$FLATBUFFERS_VERSION.tar.gz $FLATBUFFERS_DIR.tar.gz
    tar xzf $FLATBUFFERS_DIR.tar.gz
fi
cd $FLATBUFFERS_DIR
if ! command -v flatc; then
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
    make
    make install
fi
