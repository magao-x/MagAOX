#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
XRIF_COMMIT="2d0196246ecc2ae6bbc353c02959bde8b5e380b9"
#
# xrif streaming compression library
#
orgname=jaredmales
reponame=xrif
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir
git checkout $XRIF_COMMIT
mkdir -p build
cd build
cmake ..
make
make test
sudo make install
sudo ldconfig
