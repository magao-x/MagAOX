#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
if [[ -e /etc/chrony/chrony.conf ]]; then
    CHRONYCONF_PATH=/etc/chrony/chrony.conf
elif [[ -e /etc/chrony.conf ]]; then
    CHRONYCONF_PATH=/etc/chrony.conf
else
    log_error "Can't find chrony.conf. Is chrony installed?"
    exit 1
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
    log_info "Configuring chronyd for $MAGAOX_ROLE as a time minion to exao1"
    sudo tee $CHRONYCONF_PATH <<'HERE'
# chrony.conf installed by MagAO-X
# for time minion
server exao1 iburst
driftfile /var/lib/chrony/drift
makestep 1.0 3
rtcsync
HERE
elif [[ $MAGAOX_ROLE == TIC ]]; then
    log_info "Configuring chronyd as a time master for $MAGAOX_ROLE"
    sudo tee $CHRONYCONF_PATH <<'HERE'
# chrony.conf installed by MagAO-X
# for time master
server lbtntp.as.arizona.edu iburst
pool 0.centos.pool.ntp.org iburst
allow 192.168.1.0/24
driftfile /var/lib/chrony/drift
makestep 1.0 3
rtcsync
HERE
elif [[ $MAGAOX_ROLE == TOC ]]; then
    log_info "Configuring chronyd for $MAGAOX_ROLE as a time minion to exao0"
    sudo tee $CHRONYCONF_PATH <<'HERE'
# chrony.conf installed by MagAO-X
# for time minion
server exao0.as.arizona.edu iburst
driftfile /var/lib/chrony/drift
makestep 1.0 3
rtcsync
HERE
else
    log_info "Skipping chronyd setup because this isn't an instrument computer"
    exit 0
fi
sudo systemctl enable chronyd
log_info "chronyd enabled"
systemctl status chronyd || true
sudo systemctl start chronyd
log_info "chronyd started"
chronyc sources
# see https://chrony.tuxfamily.org/faq.html#_i_keep_getting_the_error_code_501_not_authorised_code
# for why -a is needed
sudo chronyc -a makestep
log_info "forced time sync"
