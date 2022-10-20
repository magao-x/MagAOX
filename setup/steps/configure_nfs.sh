#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC ]]; then
    sudo systemctl enable nfs-server.service
    sudo systemctl start nfs-server.service
    exportHosts=""
    for host in aoc rtc icc; do
        if [[ ${host,,} != ${MAGAOX_ROLE,,} ]]; then
            exportHosts="$host(ro,sync,all_squash) $exportHosts"
        fi
    done
    echo "/      $exportHosts" | sudo tee -a /etc/exports
    sudo exportfs -a

    for host in aoc rtc icc; do
        mkdir -p /srv/$host
        if ! grep -q "/srv/$host" /etc/fstab; then
            echo "$host:/ /srv/$host	nfs	noauto,x-systemd.automount,nofail,x-systemd.device-timeout=10s	0 0" | sudo tee -a /etc/fstab
        fi
    done
fi
