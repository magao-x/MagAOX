#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
if [[ ! -d /opt/EDTpdv ]]; then
    EDT_FILENAME=EDTpdv_lnx_5.5.8.2.run
    _cached_fetch https://edt.com/downloads/pdv_5-5-8-2_run/ $EDT_FILENAME
    chmod +x $EDT_FILENAME
    ./$EDT_FILENAME -- --default
    log_info "Installed EDTpdf SDK to /opt/EDTpdv"
else
    log_info "EDTpdv SDK already installed"
fi
