#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    sudo bash $0 "$@"
    exit $?
fi
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh

if [[ $MAGAOX_ROLE != "ICC" && $MAGAOX_ROLE != "RTC" ]]; then
    log_error "IRQ balance config only runs on RTC and ICC"
    exit 1
fi
if [[ ! -e /etc/sysconfig/irqbalance.bup ]]; then
    cp /etc/sysconfig/irqbalance /etc/sysconfig/irqbalance.bup
fi
cp $DIR/../../rtSetup/$MAGAOX_ROLE/irqbalance /etc/sysconfig/irqbalance
systemctl restart irqbalance
log_success "Installed /etc/sysconfig/irqbalance"
