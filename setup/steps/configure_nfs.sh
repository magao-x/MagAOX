#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
source /etc/os-release

if [[ $ID == rocky ]]; then
    nfsServiceUnit=nfs-server.service
else
    nfsServiceUnit=nfs-kernel-server.service
fi

if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC ]]; then
    sudo systemctl enable $nfsServiceUnit || exit 1
    sudo systemctl start $nfsServiceUnit || exit 1
    if command -v ufw; then
        sudo ufw allow from 192.168.0.0/24 to any port nfs || exit 1
    fi
    # /etc/systemd/system/nfs-server.service.d/override.conf
    overridePath="/etc/systemd/system/${nfsServiceUnit}.d/override.conf"
    sudo mkdir -p "/etc/systemd/system/${nfsServiceUnit}.d/" || exit 1
    echo -e "[Service]\nTimeoutStopSec=5s" | sudo tee $overridePath || exit 1
    exportHosts=""
    for host in aoc rtc icc; do
        if [[ ${host,,} != ${MAGAOX_ROLE,,} ]]; then
            exportHosts="$host(ro,sync,all_squash) $exportHosts"
        fi
    done
    exportDataLine="/data      $exportHosts"
    if ! grep -q $exportDataLine /etc/exports; then
        echo $exportDataLine | sudo tee -a /etc/exports || exit 1
        sudo exportfs -a || exit 1
        sudo systemctl reload $nfsServiceUnit || exit 1
    fi
    if [[ $MAGAOX_ROLE == AOC ]]; then
        exportBackupsLine="/mnt/backup      $exportHosts"
        if ! grep -q $exportBackupsLine /etc/exports; then
            echo $exportBackupsLine | sudo tee -a /etc/exports || exit 1
            sudo exportfs -a || exit 1
            sudo systemctl reload $nfsServiceUnit || exit 1
        fi
    fi

    for host in aoc rtc icc; do
        if [[ ${host,,} != ${MAGAOX_ROLE,,} ]]; then
            mountPath=/srv/$host/data
            sudo mkdir -p $mountPath || exit 1
            if ! grep -q $mountPath /etc/fstab; then
                echo "$host:/data $mountPath	nfs	ro,noauto,x-systemd.automount,nofail,x-systemd.device-timeout=10s,soft,timeo=30	0 0" | sudo tee -a /etc/fstab || exit 1
            fi
        fi
    done
    if [[ $MAGAOX_ROLE != AOC ]]; then
        mountPath=/srv/$host/backups
        sudo mkdir -p $mountPath || exit 1
        if ! grep -q $mountPath /etc/fstab; then
            echo "$host:/mnt/backups $mountPath	nfs	ro,noauto,x-systemd.automount,nofail,x-systemd.device-timeout=10s,soft,timeo=30	0 0" | sudo tee -a /etc/fstab || exit 1
        fi
    fi
fi
