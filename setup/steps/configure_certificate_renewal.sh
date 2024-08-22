#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

UNIT_PATH=/etc/systemd/system
sudo cp $DIR/../systemd_units/renew_certificates.service $UNIT_PATH/renew_certificates.service || exit 1
log_info "Installed unit file to $UNIT_PATH/renew_certificates.service"
sudo mkdir -p $UNIT_PATH/renew_certificates.service.d/ || exit 1
if [[ ! -e $UNIT_PATH/renew_certificates.service.d/override.conf ]]; then
    sudo tee $UNIT_PATH/renew_certificates.service.d/override.conf <<HERE
[Service]
Environment="VULTR_API_KEY=xxxxxxx"
HERE
    if [[ ! $? ]]; then
        exit_with_error "Couldn't create $UNIT_PATH/renew_certificates.service.d/override.conf"
    fi
    log_warn "Populate the Vultr API key in $UNIT_PATH/renew_certificates.service.d/override.conf to complete automatic certificate renewal config"
fi
