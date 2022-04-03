#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    sudo bash $0 "$@"
    exit $?
fi
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh

UNIT_PATH=/etc/systemd/system/
service_unit=rtc_cpuset.service

cp $DIR/../systemd_units/$service_unit $UNIT_PATH/$service_unit
log_success "Installed $service_unit to $UNIT_PATH"
systemctl daemon-reload
systemctl enable $service_unit || true
systemctl restart $service_unit || true