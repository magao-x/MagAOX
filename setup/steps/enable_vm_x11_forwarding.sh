#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

sed -E -i 's/#?X11UseLocalhost\s+yes/X11UseLocalhost no/g' /etc/ssh/sshd_config
sed -E -i 's/#?X11Forwarding\s+no/X11Forwarding yes/g' /etc/ssh/sshd_config
sudo systemctl restart sshd
