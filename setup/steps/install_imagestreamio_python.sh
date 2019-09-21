#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

orgname=magao-x
reponame=ImageStreamIO
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir

export CONDA_CHANGEPS1=false
for envname in py37 dev; do
    set +u
    conda activate $envname

    # install ImageStreamIOWrap
    cd /opt/MagAOX/source/ImageStreamIO
    git checkout dev
    python setup.py install

    #deactivate the environment before moving on
    conda deactivate
    set -u
done
