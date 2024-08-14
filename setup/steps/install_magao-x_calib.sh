#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

orgname=magao-x
reponame=calib
parentdir=/opt/MagAOX
clone_or_update_and_cd $orgname $reponame $parentdir
