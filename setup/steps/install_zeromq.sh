#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
cd /opt/MagAOX/vendor || exit 1

source /etc/os-release

if [[ $VERSION_ID == "24.04" ]]; then
    ZEROMQ_VERSION=4.3.5
else
    ZEROMQ_VERSION=4.3.4
fi

ZEROMQ_DIR=zeromq-${ZEROMQ_VERSION}
ZEROMQ_TARFILE=$ZEROMQ_DIR.tar.gz
if [[ ! -d $ZEROMQ_DIR ]]; then
    _cached_fetch https://github.com/zeromq/libzmq/releases/download/v${ZEROMQ_VERSION}/$ZEROMQ_TARFILE $ZEROMQ_TARFILE || exit 1
    tar xzf $ZEROMQ_TARFILE || exit 1
fi
cd $ZEROMQ_DIR || exit 1
./configure --enable-drafts || exit 1
make || exit 1
sudo make install || exit 1
