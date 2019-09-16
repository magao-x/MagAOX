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
	log_success "created conda env py37"
else
	log_info "py37 environment already exists"
fi

#
# Clone to create a development environment
#
if [[ $ENVS != *dev* ]]; then
	conda create --name dev --clone py37
	log_success "created conda env dev from env py37"
else
	log_info "dev environment already exists"
fi

#
# Set up auto-starting xsup Jupyter Lab instance
#
if [[ $MAGAOX_ROLE == vm ]]; then
	JUPYTERLAB_OPTIONS='--ip="0.0.0.0"'
else
	JUPYTERLAB_OPTIONS=''
fi
JUPYTERLAB_CONFIG_PATH=/opt/MagAOX/config/jupyter_notebook_config.py

# Note that there's a race condition where /vagrant isn't available yet
# when jupyterlab tries to start, so we make a copy within the VM's local
# storage.
if [[ $MAGAOX_ROLE == vm ]]; then
	cp $JUPYTERLAB_CONFIG_PATH /opt/miniconda3/envs/py37/etc/jupyter_notebook_config.py
	JUPYTERLAB_CONFIG_PATH=/opt/miniconda3/envs/py37/etc/jupyter_notebook_config.py
fi
JUPYTER_SCRIPT=/opt/miniconda3/envs/py37/bin/start_jupyterlab.sh
sudo tee $JUPYTER_SCRIPT >/dev/null <<HERE
#!/bin/bash
source /etc/profile
/opt/miniconda3/envs/py37/bin/jupyter lab --config=$JUPYTERLAB_CONFIG_PATH $JUPYTERLAB_OPTIONS
HERE
chmod +x $JUPYTER_SCRIPT
UNIT_PATH=/etc/systemd/system/

if [[ $MAGAOX_ROLE != RTC && $MAGAOX_ROLE != ICC ]]; then
	cp /opt/MagAOX/config/jupyterlab.service $UNIT_PATH/jupyterlab.service
	log_success "Installed jupyterlab.service to $UNIT_PATH"
	if [[ $MAGAOX_ROLE == vm ]]; then
		sed -iE "s/xsup/vagrant/g" $UNIT_PATH/jupyterlab.service
		log_info "Rewrote service for vagrant"
	fi
	systemctl enable jupyterlab
	log_success "Enabled jupyterlab service"
	systemctl start jupyterlab
	log_success "Started jupyterlab service"
fi
