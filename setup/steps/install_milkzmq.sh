#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -o pipefail
# install milkzmq
MILKZMQ_COMMIT_ISH=master
orgname=jaredmales
reponame=milkzmq
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir

git checkout $MILKZMQ_COMMIT_ISH
make || exit 1
sudo make install || exit 1
