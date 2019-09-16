#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/vendor/andor
sudo patch -Np1 < unattended_install.patch || true
sudo bash install_andor
log_info "install_andor complete"
