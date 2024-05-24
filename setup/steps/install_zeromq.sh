#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/vendor

source /etc/os-release

if [[ $VERSION_ID == "24.04" ]]; then
    ZEROMQ_VERSION=4.3.5
else
    ZEROMQ_VERSION=4.3.4
fi

ZEROMQ_DIR=zeromq-${ZEROMQ_VERSION}
ZEROMQ_TARFILE=$ZEROMQ_DIR.tar.gz
if [[ ! -d $ZEROMQ_DIR ]]; then
    _cached_fetch https://github.com/zeromq/libzmq/releases/download/v${ZEROMQ_VERSION}/$ZEROMQ_TARFILE $ZEROMQ_TARFILE
    tar xzf $ZEROMQ_TARFILE
fi
cd $ZEROMQ_DIR
./configure --enable-drafts
make
sudo make install
