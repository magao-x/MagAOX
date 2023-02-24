#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILDING_KERNEL_STUFF=1  # disable loading devtoolset-7 for agreement w/ kernel gcc
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/vendor
if [[ ! -d /opt/EDTpdv ]]; then
    EDT_VERSION="5.6.2.0"
    EDT_FILENAME=EDTpdv_lnx_$EDT_VERSION.run
    _cached_fetch https://edt.com/downloads/pdv_${EDT_VERSION//./-}_run/ $EDT_FILENAME
    chmod +x $EDT_FILENAME
    ./$EDT_FILENAME -- --default
    log_info "Installed EDTpdv SDK to /opt/EDTpdv"
else
    log_info "EDTpdv SDK already installed"
fi

sudo mv /opt/EDTpdv/version /opt/EDTpdv/version.txt 2>/dev/null || true
