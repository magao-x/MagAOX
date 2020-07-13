#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

RTC_DEST_PORT=6666
ICC_DEST_PORT=6669
NETCONSOLE_PORTS=$RTC_DEST_PORT,rtc:$ICC_DEST_PORT,icc
# Interface to bind to for log *listener* (AOC on instrument LAN)
NETCONSOLE_BIND_IP=192.168.0.10

if [[ $MAGAOX_ROLE == AOC ]]; then
    UNIT_PATH=/etc/systemd/system/
    if [[ ! -e $UNIT_PATH/netconsole_logger.service ]]; then
        sudo cp /opt/MagAOX/config/netconsole_logger.service $UNIT_PATH/netconsole_logger.service
        log_info "Installed unit file to $UNIT_PATH/netconsole_logger.service"
        OVERRIDE_PATH=$UNIT_PATH/netconsole.service.d/
        sudo mkdir -p $OVERRIDE_PATH
        sudo tee $OVERRIDE_PATH/override.conf <<HERE
[Service]
Environment="NETCONSOLE_PORTS=$NETCONSOLE_PORTS"
Environment="NETCONSOLE_BIND_IP=$NETCONSOLE_BIND_IP
HERE

    fi
    sudo mkdir $NETCONSOLE_LOG_DIR
    sudo chown :magaox $NETCONSOLE_LOG_DIR
    sudo chmod g+w $NETCONSOLE_LOG_DIR
    sudo systemctl enable netconsole_logger.service || true
    sudo systemctl restart netconsole_logger.service || true
elif [[ $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC ]]; then
    echo 'netconsole' | sudo tee /etc/modules-load.d/netconsole.conf
    if [[ $MAGAOX_ROLE == ICC ]]; then
        netconsole_options="options netconsole netconsole=6665@192.168.0.12/enx3497f65a200d,$ICC_DEST_PORT@192.168.0.10/2c:fd:a1:c6:1d:de"
        log_info "Logging from ICC kernel console to UDP $ICC_DEST_PORT on AOC"
    elif [[ $MAGAOX_ROLE == RTC ]]; then
        netconsole_options="options netconsole netconsole=6665@192.168.0.11/enx2cfda1c6db1a,$RTC_DEST_PORT@192.168.0.10/2c:fd:a1:c6:1d:de"
        log_info "Logging from RTC kernel console to UDP $RTC_DEST_PORT on AOC"
    fi
    echo $netconsole_options | sudo tee /etc/modprobe.d/netconsole.conf
    log_success 'netconsole configured'
    modprobe netconsole
    log_info 'netconsole module loaded'
fi