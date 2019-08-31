#!/bin/bash

#
# This script is still a bit broken and to be run with the '-i' bash flag 
# in order to activate a conda environment
#
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

#
# Create the standard MagAOX user python environment
#
ENVS=$(conda env list)
if [[ $ENVS != *py37* ]]; then
	conda env create -f /opt/MagAOX/config/conda_env_py37.yml
else
	echo "py37 environment already exists!"
fi

# This line breaks.
#eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
conda activate py37

# install ImageStreamWrapIO (requires cacao to be installed)
PACKAGES=$(conda list)
if [[ $PACKAGES != *imagestreamiowrap* ]]; then
	# setup.py doesn't realize cmake != cmake3 on CentOS
	if [[ ! -f /opt/miniconda3/envs/py37/bin/cmake ]]; then
		ln -s /usr/bin/cmake3 /opt/miniconda3/envs/py37/bin/cmake
	fi
	export CUDA_ROOT=$CUDADIR
	cd /opt/MagAOX/source/cacao.old/src/ImageStreamIO #change me
	python setup.py install
	# clean up cmake
	rm /opt/miniconda3/envs/py37/bin/cmake
else
	echo "imagestreamiowrap already installed in py37 environment!"
fi
#deactivate the environment before moving on
conda deactivate

#
# Clone to create a development environment
#
if [[ $ENVS != *dev* ]]; then
	conda create --name dev --clone py37
else
	echo "dev environment already exists!"
fi
