#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

MAGPYX_COMMIT_ISH=master
orgname=magao-x
reponame=magpyx
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir
git checkout $MAGPYX_COMMIT_ISH

pip install -e .
python -c 'import magpyx'
