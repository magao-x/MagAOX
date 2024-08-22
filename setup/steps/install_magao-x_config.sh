#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

orgname=magao-x
reponame=config
parentdir=/opt/MagAOX/
clone_or_update_and_cd $orgname $reponame $parentdir || exit 1
