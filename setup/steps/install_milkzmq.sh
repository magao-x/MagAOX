#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

# install milkzmq
MILKZMQ_COMMIT_ISH=master
if [[ ! -d ./milkzmq ]]; then
    git clone https://github.com/jaredmales/milkzmq.git milkzmq
    cd ./milkzmq
else
    cd ./milkzmq
    git pull origin master
fi
git config core.sharedRepository group
git checkout $MILKZMQ_COMMIT_ISH
make
sudo make install
