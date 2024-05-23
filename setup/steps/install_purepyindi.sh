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

cd $parentdir/$reponame
sudo -H /opt/conda/bin/pip install -e .[all]
python -c 'import purepyindi'
