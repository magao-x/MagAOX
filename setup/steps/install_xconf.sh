#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

COMMIT_ISH=main
orgname=xwcl
reponame=xconf
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir
git checkout $COMMIT_ISH

cd $parentdir/$reponame
sudo /opt/conda/bin/pip install -e .[all]
python -c 'import xconf'
