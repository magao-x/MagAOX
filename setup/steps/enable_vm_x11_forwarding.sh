#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

sudo sed -E -i 's/#?X11UseLocalhost\s+yes//g' /etc/ssh/sshd_config
if ! sudo grep -q 'X11UseLocalhost no' /etc/ssh/sshd_config; then
    echo 'X11UseLocalhost no' | sudo tee -a /etc/ssh/sshd_config
fi
sed -E -i 's/#?X11Forwarding\s+no//g' /etc/ssh/sshd_config
if ! sudo grep -q 'X11Forwarding no' /etc/ssh/sshd_config; then
    echo 'X11Forwarding no' | sudo tee -a /etc/ssh/sshd_config
fi
sudo systemctl restart sshd
########################################################################
### xauth is installed elsewhere; left commented out here in case it is
### needed for diagnostics later
########################################################################
## Necessary for forwarding GUIs from the VM to the host
## Shell variables $ID and $VERSION_ID were sourced from /etc/os-release
##   in _common.sh above, so we can detect which distribution we're on
#if [[ $ID == ubuntu ]]; then
#    sudo NEEDRESTART_SUSPEND=yes apt install -y xauth
#elif [[ $ID == rocky ]]; then
#    sudo yum install -y xorg-x11-xauth
#fi
