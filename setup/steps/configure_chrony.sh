#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
if [[ -e /etc/chrony/chrony.conf ]]; then
    CHRONYCONF_PATH=/etc/chrony/chrony.conf
elif [[ -e /etc/chrony.conf ]]; then
    CHRONYCONF_PATH=/etc/chrony.conf
fi

if [[ $MAGAOX_ROLE == AOC ]]; then
    log_info "Configuring chronyd as a time master for $MAGAOX_ROLE"
    sudo tee $CHRONYCONF_PATH <<'HERE'
# chrony.conf installed by MagAO-X
# for time master
server lbtntp.as.arizona.edu iburst
server ntp1.lco.cl iburst
server ntp2.lco.cl iburst
pool 0.centos.pool.ntp.org iburst
allow 192.168.0.0/24
driftfile /var/lib/chrony/drift
makestep 1.0 3
rtcsync
HERE
elif [[ $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC ]]; then
    log_info "Configuring chronyd as a time minion for $MAGAOX_ROLE"
    sudo tee $CHRONYCONF_PATH <<'HERE'
# chrony.conf installed by MagAO-X
# for time minion
server exao1 iburst
driftfile /var/lib/chrony/drift
makestep 1.0 3
rtcsync
HERE
else
    log_info "Skipping chronyd setup because this isn't an instrument computer"
fi
sudo systemctl enable chronyd
log_info "chronyd enabled"
systemctl status chronyd || true
sudo systemctl start chronyd
log_info "chronyd started"
chronyc sources
chronyc tracking
# see https://chrony.tuxfamily.org/faq.html#_i_keep_getting_the_error_code_501_not_authorised_code
# for why -a is needed
sudo chronyc -a makestep
log_info "forced time sync"
