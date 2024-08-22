#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
XRIF_COMMIT="2d0196246ecc2ae6bbc353c02959bde8b5e380b9"
#
# xrif streaming compression library
#
orgname=jaredmales
reponame=xrif
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir || exit 1
git checkout $XRIF_COMMIT || exit 1
mkdir -p build || exit 1
cd build || exit 1
cmake .. || exit 1
make || exit 1
make test || exit 1
sudo make install || exit 1
sudo ldconfig || exit 1
