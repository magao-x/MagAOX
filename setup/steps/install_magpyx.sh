#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

orgname=magao-x
reponame=magpyx
if [[ $MAGAOX_ROLE == vm ]]; then
    parentdir=/vagrant/vm
else
    parentdir=/opt/MagAOX/source
fi
clone_or_update_and_cd $orgname $reponame $parentdir

export CONDA_CHANGEPS1=false
for envname in py37 dev; do
    set +u
    conda activate $envname

    PACKAGES=$(conda list)
    if [[ $PACKAGES != *magpyx* ]]; then
        cd $parentdir/magpyx
        pip install -e .
    else
        echo "magpyx already installed in $envname environment!"
    fi
    python -c "import magpyx"
    #deactivate the environment before moving on
    conda deactivate
    set -u
done
