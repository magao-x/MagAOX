#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -ue
XRIF_COMMIT="fdfce2bff20f22fa5d965ed290a8e0c4f9ff64d5"
XRIF_DIR="./xrif"

#
# xrif streaming compression library
#
yum install -y check-devel subunit-devel

if [[ ! -d $XRIF_DIR ]]; then
    git clone https://github.com/jaredmales/xrif.git $XRIF_DIR
fi
cd $XRIF_DIR
git checkout $XRIF_COMMIT
mkdir -p build
cd build
cmake3 ..
make
make test
make install
ldconfig
