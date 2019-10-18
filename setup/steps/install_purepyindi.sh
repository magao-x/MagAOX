#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

PUREPYINDI_COMMIT_ISH=master
orgname=magao-x
reponame=purepyindi
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir
git checkout $PUREPYINDI_COMMIT_ISH

for envname in py37 dev; do
    set +u; conda activate $envname; set -u
    cd $parentdir/$reponame
    pip install -e .[all]
    python -c 'import purepyindi'
done