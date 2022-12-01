#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
source /etc/os-release

if [[ $ID == centos ]]; then
    nfsServiceUnit=nfs-server.service
else
    nfsServiceUnit=nfs-kernel-server.service
fi

if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC ]]; then
    sudo systemctl enable $nfsServiceUnit
    sudo systemctl start $nfsServiceUnit
    if command -v ufw; then
        sudo ufw allow from 192.168.0.0/24 to any port nfs
    fi
    # /etc/systemd/system/nfs-server.service.d/override.conf
    overridePath="/etc/systemd/system/${nfsServiceUnit}.d/override.conf"
    sudo mkdir -p "/etc/systemd/system/${nfsServiceUnit}.d/"
    echo -e "[Service]\nTimeoutStopSec=5s" | sudo tee $overridePath
    exportHosts=""
    for host in aoc rtc icc; do
        if [[ ${host,,} != ${MAGAOX_ROLE,,} ]]; then
            exportHosts="$host(ro,sync,all_squash) $exportHosts"
        fi
    done
    if ! grep -q "$exportHosts" /etc/exports; then
        echo "/data      $exportHosts" | sudo tee -a /etc/exports
        sudo exportfs -a
        sudo systemctl reload $nfsServiceUnit
    fi

    for host in aoc rtc icc; do
        if [[ ${host,,} != ${MAGAOX_ROLE,,} ]]; then
            mountPath=/srv/$host/data
            sudo mkdir -p $mountPath
            if ! grep -q $mountPath /etc/fstab; then
                echo "$host:/data $mountPath	nfs	noauto,x-systemd.automount,nofail,x-systemd.device-timeout=10s	0 0" | sudo tee -a /etc/fstab
            fi
        fi
    done
fi
