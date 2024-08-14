#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
ZLIB_COMMIT="develop-20240718-drbitboy"
#
# xrif streaming compression library
#
orgname=drbitboy
reponame=zlib-drbitboy
parentdir=/opt/MagAOX/vendor
destdir=""
clone_or_update_and_cd $orgname $reponame $parentdir $destdir
git checkout $ZLIB_COMMIT
make distclean && ./configure && make test && sudo make install
