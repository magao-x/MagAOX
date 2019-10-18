#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

IMAGESTREAMIO_COMMIT_ISH=dev
orgname=magao-x
reponame=ImageStreamIO
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir
git checkout $IMAGESTREAMIO_COMMIT_ISH

for envname in py37 dev; do
    conda activate $envname
    cd $parentdir/$reponame
    python setup.py install
    python -c 'import ImageStreamIOWrap'
done