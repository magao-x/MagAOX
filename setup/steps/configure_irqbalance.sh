#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    sudo -H bash $0 "$@"
    exit $?
fi
set -uo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh

if [[ $MAGAOX_ROLE != "ICC" && $MAGAOX_ROLE != "RTC" ]]; then
    log_error "IRQ balance config only runs on RTC and ICC"
    exit 1
fi
if [[ ! -e /etc/sysconfig/irqbalance.bup ]]; then
    cp /etc/sysconfig/irqbalance /etc/sysconfig/irqbalance.bup || exit 1
fi

# Install irqbalance config to use policyscript
cp "$DIR/../../rtSetup/$MAGAOX_ROLE/irqbalance" /etc/sysconfig/irqbalance || exit 1

# Install policyscript
if [[ ! -e /usr/local/bin/irqbalance_policyscript ]]; then
    install --owner root --mode 0555 \
        "$DIR/../../rtSetup/$MAGAOX_ROLE/irqbalance_policyscript" \
        /usr/local/bin/irqbalance_policyscript || exit 1
fi

# Note that SELinux audit events are still logged, but permissive is required
# to get the policyscript to run from the irqbalance service's context.
semanage permissive -a irqbalance_t || exit 1
systemctl restart irqbalance || exit 1
log_success "Installed /etc/sysconfig/irqbalance"
