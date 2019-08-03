#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

# install milkzmq
MILKZMQ_COMMIT=f427ed35986b3956a937adef349820d1d3a626f0
if [[ ! -d ./milkzmq ]]; then
    git clone https://github.com/jaredmales/milkzmq.git milkzmq
    cd ./milkzmq
else
    cd ./milkzmq
    git fetch
fi
git config core.sharedRepository group
git checkout $MILKZMQ_COMMIT
make
sudo make install
