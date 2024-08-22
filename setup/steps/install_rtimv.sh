#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
RTIMV_COMMIT_ISH=dev
orgname=jaredmales
reponame=rtimv
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir || exit 1
git checkout $RTIMV_COMMIT_ISH || exit 1
make || exit 1
sudo make install || exit 1
