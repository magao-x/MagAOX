#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

# install cppzmq (dependency of milkzmq)
CPPZMQ_COMMIT=213da0b04ae3b4d846c9abc46bab87f86bfb9cf4
if [[ ! -d cppzmq ]]; then
    git clone https://github.com/zeromq/cppzmq.git
    cd cppzmq
else
    cd cppzmq
    git fetch
fi
git checkout $CPPZMQ_COMMIT
sudo cp *.hpp /usr/local/include/
