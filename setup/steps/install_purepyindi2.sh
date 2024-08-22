#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

PUREPYINDI2_COMMIT_ISH=main
orgname=xwcl
reponame=purepyindi2
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir || exit 1
git checkout $PUREPYINDI2_COMMIT_ISH || exit 1

cd $parentdir/$reponame || exit 1
sudo -H /opt/conda/bin/pip install -e .[all] || exit 1
python -c 'import purepyindi2' || exit 1
