#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
if [[ ! -d /opt/EDTpdv ]]; then
    EDT_VERSION="5.5.8.2"
    EDT_FILENAME=EDTpdv_lnx_$EDT_VERSION.run
    _cached_fetch https://edt.com/downloads/pdv_${EDT_VERSION//./-}_run/ $EDT_FILENAME
    chmod +x $EDT_FILENAME
    ./$EDT_FILENAME -- --default
    log_info "Installed EDTpdv SDK to /opt/EDTpdv"
else
    log_info "EDTpdv SDK already installed"
fi
