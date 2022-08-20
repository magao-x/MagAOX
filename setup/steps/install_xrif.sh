#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
XRIF_COMMIT="2d0196246ecc2ae6bbc353c02959bde8b5e380b9"
XRIF_DIR="./xrif"

#
# xrif streaming compression library
#

if [[ ! -d $XRIF_DIR ]]; then
    git clone https://github.com/jaredmales/xrif.git $XRIF_DIR
fi
cd $XRIF_DIR
git config core.sharedRepository group
git checkout $XRIF_COMMIT
mkdir -p build
cd build
cmake ..
make
make test
make install
ldconfig
