#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILDING_KERNEL_STUFF=1  # disable loading devtoolset-7 for agreement w/ kernel gcc
source $DIR/../_common.sh
set -uo pipefail
cd /opt/MagAOX/vendor/andor
sudo -H patch -Np1 < unattended_install.patch || true
sudo -H bash install_andor || exit 1
log_info "install_andor complete"
