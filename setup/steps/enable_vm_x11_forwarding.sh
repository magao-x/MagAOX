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
