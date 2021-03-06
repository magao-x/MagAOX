#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

# install cppzmq (dependency of milkzmq)
CPPZMQ_COMMIT=213da0b04ae3b4d846c9abc46bab87f86bfb9cf4
if [[ ! -d ./cppzmq-$CPPZMQ_COMMIT ]]; then
    _cached_fetch https://github.com/zeromq/cppzmq/archive/$CPPZMQ_COMMIT.tar.gz cppzmq-$CPPZMQ_COMMIT.tar.gz
    tar xzf cppzmq-$CPPZMQ_COMMIT.tar.gz
fi
cd ./cppzmq-$CPPZMQ_COMMIT
sudo cp *.hpp /usr/local/include/
