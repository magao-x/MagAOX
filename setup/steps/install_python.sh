#!/bin/bash
# If not started as root, sudo yourself
if [[ "$EUID" != 0 ]]; then
    sudo bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
MINICONDA_VERSION="3-4.7.10"
#
# MINICONDA
#
cd /opt/MagAOX/vendor
if [[ ! -d /opt/miniconda3 ]]; then
    MINICONDA_INSTALLER="Miniconda$MINICONDA_VERSION-Linux-x86_64.sh"
    _cached_fetch "https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER" $MINICONDA_INSTALLER
    bash $MINICONDA_INSTALLER -b -p /opt/miniconda3
    # Set environment variables for miniconda
    cat << 'EOF' | tee /etc/profile.d/miniconda.sh
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
    cat << 'EOF' | tee /opt/miniconda3/.condarc
channels:
  - conda-forge
  - defaults
changeps1: false
EOF
fi

# Make sure the newly created shell tidbit
# is sourced before we try to use `conda` below
source /etc/profile.d/miniconda.sh

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

if [[ $MAGAOX_ROLE != RTC && $MAGAOX_ROLE != ICC && $MAGAOX_ROLE != ci ]]; then
	cp /opt/MagAOX/config/jupyterlab.service $UNIT_PATH/jupyterlab.service
	log_success "Installed jupyterlab.service to $UNIT_PATH"
	if [[ $MAGAOX_ROLE == vm ]]; then
		sed -iE "s_WorkingDirectory=/home/xsup_WorkingDirectory=/vagrant_g" $UNIT_PATH/jupyterlab.service
		sed -iE "s/xsup/vagrant/g" $UNIT_PATH/jupyterlab.service
		log_info "Rewrote service for vagrant"
	fi
	systemctl enable jupyterlab
	log_success "Enabled jupyterlab service"
	systemctl start jupyterlab
	log_success "Started jupyterlab service"
fi


# set group and permissions such that only magaox-dev has write access
chgrp -R magaox-dev /opt/miniconda3
chmod -R g=rwX /opt/miniconda3
