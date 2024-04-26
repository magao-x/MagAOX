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
mamba env update -f $DIR/../conda_env_pinned_$(uname -i).yml || exit_with_error "Failed to install or update packages using pinned versions. Update the env manually with the base specification and update the pinned versions if possible."
source /etc/os-release
if [[ ( $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ci ) && ( $ID == "centos" ) ]]; then
	mamba install -y qt=5 qwt
fi

#
# Set up auto-starting xsup Jupyter Notebook instance
#
if [[ $MAGAOX_ROLE == vm ]]; then
	NOTEBOOK_OPTIONS='--ip="0.0.0.0"'
else
	NOTEBOOK_OPTIONS=''
fi
NOTEBOOK_CONFIG_PATH=$DIR/../jupyter_notebook_config.py

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
    sudo systemctl stop jupyterlab || true
	sudo rm $UNIT_PATH/jupyterlab.service
fi

if [[ $MAGAOX_ROLE != ci ]]; then
	sudo cp $DIR/../systemd_units/jupyternotebook.service $UNIT_PATH/jupyternotebook.service
	log_success "Installed jupyternotebook.service to $UNIT_PATH"

	# Due to SystemD nonsense, WorkingDirectory must be a directory and not a symbolic link
	# and due to MagAO-X nonsense those directories are different if the role has a /data array
	# ...but at least there is 'override.conf'
	OVERRIDE_PATH=$UNIT_PATH/jupyternotebook.service.d/
	sudo mkdir -p $OVERRIDE_PATH
	log_info "Made $OVERRIDE_PATH for override"
	overrideFileDest=$OVERRIDE_PATH/override.conf
	overrideFile=/tmp/jupyternotebook_$(date +"%s")
	echo "[Service]" > $overrideFile
	workingDir=/home/xsup/data
	if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == TIC ]]; then
		workingDir=/data/users/xsup
	fi
	echo "WorkingDirectory=$workingDir" >> $overrideFile
	if [[ -e $overrideFileDest ]]; then
		if ! diff $overrideFile $overrideFileDest; then
			exit_with_error "Existing $overrideFile does not match $overrideFileDest"
		fi
	else
		sudo mv $overrideFile $overrideFileDest
	fi

	sudo systemctl daemon-reload
	
	sudo systemctl enable jupyternotebook
	log_success "Enabled jupyternotebook service"
	sudo systemctl start jupyternotebook
	log_success "Started jupyternotebook service"
fi
