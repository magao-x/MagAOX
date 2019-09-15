#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

orgname=magao-x
reponame=purepyindi
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir

export CONDA_CHANGEPS1=false
for envname in py37 dev; do
    conda activate $envname

    PACKAGES=$(conda list)
    if [[ $PACKAGES != *purepyindi* ]]; then
        cd /opt/MagAOX/source/purepyindi
        pip install -e .[all]
    else
        echo "purepyindi already installed in $envname environment!"
    fi
    #deactivate the environment before moving on
    conda deactivate
done
