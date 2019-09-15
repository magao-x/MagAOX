#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
MINICONDA_VERSION="3-4.7.10"
#
# MINICONDA
#
if [[ ! -d /opt/miniconda3 ]]; then
    MINICONDA_INSTALLER="Miniconda$MINICONDA_VERSION-Linux-x86_64.sh"
    _cached_fetch "https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER" $MINICONDA_INSTALLER
    sudo bash $MINICONDA_INSTALLER -b -p /opt/miniconda3
    # Set environment variables for miniconda
    cat << 'EOF' | sudo tee /etc/profile.d/miniconda.sh
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/opt/miniconda3/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=false conda activate base
else
    \export PATH="/opt/miniconda3/bin:$PATH"
fi
# activate the default MagAO-X user env if it exists
ENVS=$(conda env list)
if [[ $ENVS = *py37* ]]; then
    conda activate py37
fi
EOF
    cat << 'EOF' | sudo tee /opt/miniconda3/.condarc
channels:
  - conda-forge
  - defaults
changeps1: false
EOF
fi
set +u; source /etc/profile.d/miniconda.sh; set -u
# set group and permissions such that only magaox-dev has write access
sudo chgrp -R magaox-dev /opt/miniconda3
sudo chmod -R g=rwx /opt/miniconda3
# TO DO: Clone git repos for any MagAO-X python utils and install
echo "You may need to run \"source /etc/profile.d/miniconda.sh\" before conda is ready to use."
