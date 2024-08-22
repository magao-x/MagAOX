#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

orgname=magao-x
reponame=testbed_calib
parentdir=/opt/MagAOX/
destdir=$parentdir/calib
clone_or_update_and_cd $orgname $reponame $parentdir $destdir || exit 1
