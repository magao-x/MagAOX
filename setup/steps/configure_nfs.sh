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
    exportHosts=""
    for host in aoc rtc icc; do
        if [[ ${host,,} != ${MAGAOX_ROLE,,} ]]; then
            exportHosts="$host(ro,sync,all_squash) $exportHosts"
        fi
    done
    if ! grep -q "$exportHosts" /etc/exports; then
        echo "/      $exportHosts" | sudo tee -a /etc/exports
        sudo exportfs -a
        sudo systemctl restart $nfsServiceUnit
    fi

    for host in aoc rtc icc; do
        if [[ ${host,,} != ${MAGAOX_ROLE,,} ]]; then
            sudo mkdir -p /srv/$host
            if ! grep -q "/srv/$host" /etc/fstab; then
                echo "$host:/ /srv/$host	nfs	noauto,x-systemd.automount,nofail,x-systemd.device-timeout=10s	0 0" | sudo tee -a /etc/fstab
            fi
        fi
    done
fi
