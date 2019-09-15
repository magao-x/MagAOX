#!/bin/bash
#
# This script is still a bit broken and to be run with the '-i' bash flag
# in order to activate a conda environment
#
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

#
# Create the standard MagAOX user python environment
#
ENVS=$(conda env list)
if [[ $ENVS != *py37* ]]; then
	conda env create -f /opt/MagAOX/config/conda_env_py37.yml
else
	echo "py37 environment already exists!"
fi

#
# Clone to create a development environment
#
if [[ $ENVS != *dev* ]]; then
	conda create --name dev --clone py37
else
	echo "dev environment already exists!"
fi
