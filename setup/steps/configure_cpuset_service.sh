#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    sudo bash $0 "$@"
    exit $?
fi
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh

if [[ $MAGAOX_ROLE != "ICC" && $MAGAOX_ROLE != "RTC" ]]; then
    log_error "cpuset scripts only run on RTC and ICC"
    exit 1
fi

UNIT_PATH=/etc/systemd/system/
service_unit=${MAGAOX_ROLE}_cpuset.service

cp $DIR/../systemd_units/$service_unit $UNIT_PATH/$service_unit
log_success "Installed $service_unit to $UNIT_PATH"
systemctl daemon-reload
systemctl enable $service_unit || true
systemctl restart $service_unit || true