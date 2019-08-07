#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
MINICONDA_VERSION="3-4.7.10"
#
# MINICONDA
#
MINICONDA_INSTALLER="Miniconda$MINICONDA_VERSION-Linux-x86_64.sh"
if [[ ! -e $MINICONDA_INSTALLER ]]; then
    curl -OL "https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER"
fi
sudo bash $MINICONDA_INSTALLER -b -p /opt/miniconda3
# Set environment variables for miniconda
echo "__conda_setup=\"\$(CONDA_REPORT_ERRORS=false '/opt/miniconda3/bin/conda' shell.bash hook 2> /dev/null)\"
if [ \$? -eq 0 ]; then
    \eval \"\$__conda_setup\"
else
    if [ -f \"/opt/miniconda3/etc/profile.d/conda.sh\" ]; then
        . \"/opt/miniconda3/etc/profile.d/conda.sh\"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH=\"/opt/miniconda3/bin:\$PATH\"
    fi
fi
unset __conda_setup" > /etc/profile.d/miniconda.sh
source /etc/profile.d/miniconda.sh
# set group and permissions such that only magaox-dev has write access
sudo chgrp -R magaox-dev /opt/miniconda3
sudo chmod -R g=rwx /opt/miniconda3
# TO DO: Build the python environment from an environment.yml file
# TO DO: Manually install pyImageStreamIO in cacao/src (need to force cmake3 to run as cmake)
# TO DO: Clone git repos for any MagAO-X python utils and install
echo "You may need to run \"source /etc/profile.d/miniconda.sh\" before conda is ready to use."
