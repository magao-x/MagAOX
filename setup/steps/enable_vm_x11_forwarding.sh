#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

sed -E -i 's/#?X11UseLocalhost\s+yes/X11UseLocalhost no/g' /etc/ssh/sshd_config
sed -E -i 's/#?X11Forwarding\s+no/X11Forwarding yes/g' /etc/ssh/sshd_config
sudo systemctl restart sshd

# Necessary for forwarding GUIs from the VM to the host
source /etc/os-release # Defines $ID and $VERSION_ID so we can detect which distribution we're on
if [[ $ID == ubuntu ]]; then
    sudo apt install -y xauth
elif [[ $ID == centos ]]; then
    sudo yum install -y xorg-x11-xauth
fi
