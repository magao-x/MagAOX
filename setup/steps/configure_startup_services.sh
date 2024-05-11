#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    sudo -H bash $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh

UNIT_PATH=/etc/systemd/system
for service_unit in make_cpusets.service move_irqs.service; do
    cp $DIR/../systemd_units/$service_unit $UNIT_PATH/$service_unit
    log_success "Installed $service_unit to $UNIT_PATH"
    systemctl daemon-reload
    systemctl enable $service_unit || true
done

# OVERRIDE_PATH=$UNIT_PATH/xctrl_startup.service.d
# mkdir -p $OVERRIDE_PATH
# echo "[Service]" > $OVERRIDE_PATH/override.conf
# echo "Environment=\"MAGAOX_ROLE=$MAGAOX_ROLE\"" >> $OVERRIDE_PATH/override.conf
# if [[ $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC ]]; then
# echo "[Unit]" >> $OVERRIDE_PATH/override.conf
# echo "After=edtinit.service" >> $OVERRIDE_PATH/override.conf
# fi
# systemctl daemon-reload
# log_success "Added MAGAOX_ROLE to $OVERRIDE_PATH/override.conf"
