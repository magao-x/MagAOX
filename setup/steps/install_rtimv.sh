#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

orgname=jaredmales
reponame=rtimv
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir
make install
