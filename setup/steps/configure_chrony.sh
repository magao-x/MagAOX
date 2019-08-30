#!/bin/bash
if [[ -e /etc/chrony/chrony.conf ]]; then
    CHRONYCONF_PATH=/etc/chrony/chrony.conf
elif [[ -e /etc/chrony.conf ]]; then
    CHRONYCONF_PATH=/etc/chrony.conf
fi
read -d '' MASTER_CONF <<'HERE'
server lbtntp.as.arizona.edu iburst
server ntp1.lco.cl iburst
server ntp2.lco.cl iburst
pool 0.centos.pool.ntp.org iburst
allow 192.168.1.0/24
driftfile /var/lib/chrony/drift
makestep 1.0 3
rtcsync
HERE
read -d '' MINION_CONF <<'HERE'
server exao1 iburst
driftfile /var/lib/chrony/drift
makestep 1.0 3
rtcsync
HERE

if [[ $MAGAOX_ROLE == aoc ]]; then
    log_info "Configuring chronyd as a time master for $MAGAOX_ROLE"
    echo $MINON_CONF | sudo tee $CHRONYCONF_PATH
elif [[ $MAGAOX_ROLE == icc || $MAGAOX_ROLE == rtc ]]; then
    log_info "Configuring chronyd as a time minion for $MAGAOX_ROLE"
    echo $MINON_CONF | sudo tee $CHRONYCONF_PATH
else
    log_info "Skipping chronyd setup because this isn't an instrument computer"
fi
sudo systemctl enable chrony
systemctl status chronyd
chronyc sources
chronyc tracking
# see https://chrony.tuxfamily.org/faq.html#_i_keep_getting_the_error_code_501_not_authorised_code
# for why -a is needed
sudo chronyc -a makestep
