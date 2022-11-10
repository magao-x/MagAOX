#!/bin/bash
# If not started as root, sudo yourself
if [[ "$EUID" != 0 ]]; then
    sudo bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

#
# Install the standard MagAOX user python environment
#
mamba env update -qf $DIR/../conda_env_base.yml

#
# Set up auto-starting xsup Jupyter Notebook instance
#
if [[ $MAGAOX_ROLE == vm ]]; then
	NOTEBOOK_OPTIONS='--ip="0.0.0.0"'
else
	NOTEBOOK_OPTIONS=''
fi
NOTEBOOK_CONFIG_PATH=$DIR/../jupyter_notebook_config.py

# Note that there's a race condition where /vagrant isn't available yet
# when jupyter tries to start, so we make a copy within the VM's local
# storage.
if [[ $MAGAOX_ROLE == vm ]]; then
	cp $NOTEBOOK_CONFIG_PATH /opt/conda/etc/jupyter_notebook_config.py
	NOTEBOOK_CONFIG_PATH=/opt/conda/etc/jupyter_notebook_config.py
fi
JUPYTER_SCRIPT=/opt/conda/bin/start_notebook.sh
sudo tee $JUPYTER_SCRIPT >/dev/null <<HERE
#!/bin/bash
source /etc/profile
/opt/conda/bin/jupyter notebook --config=$NOTEBOOK_CONFIG_PATH $NOTEBOOK_OPTIONS
HERE
chmod +x $JUPYTER_SCRIPT
UNIT_PATH=/etc/systemd/system/

# clean up old files if they exist
if [[ -e $UNIT_PATH/jupyterlab.service ]]; then
    systemctl stop jupyterlab || true
	rm $UNIT_PATH/jupyterlab.service
fi

if [[ $MAGAOX_ROLE != ci && $VM_KIND != wsl ]]; then
	if [[ $MAGAOX_ROLE == AOC ]]; then
		cp $DIR/../systemd_units/lookyloo.service $UNIT_PATH/lookyloo.service
		log_success "Installed lookyloo.service to $UNIT_PATH"
	fi
	
	cp $DIR/../systemd_units/jupyternotebook.service $UNIT_PATH/jupyternotebook.service
	log_success "Installed jupyternotebook.service to $UNIT_PATH"
	if [[ $MAGAOX_ROLE == vm ]]; then
		sed -iE "s_WorkingDirectory=/home/xsup/data_WorkingDirectory=/vagrant_g" $UNIT_PATH/jupyternotebook.service
	    sed -iE "s/xsup/vagrant/g" $UNIT_PATH/jupyternotebook.service
		log_info "Rewrote service for vagrant"
	fi
	
	systemctl daemon-reload
	
	systemctl enable jupyternotebook
	log_success "Enabled jupyternotebook service"
	systemctl start jupyternotebook
	log_success "Started jupyternotebook service"
	if [[ $MAGAOX_ROLE == AOC ]]; then
		systemctl enable lookyloo
		log_success "Enabled lookyloo service"
		systemctl start lookyloo
		log_success "Started lookyloo service"
	fi
fi
