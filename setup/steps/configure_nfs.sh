#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

sudo yum -y install nfs-utils
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC ]]; then
    sudo systemctl enable nfs-server.service
    sudo systemctl start nfs-server.service
    echo <<<'HERE' | sudo tee /etc/exports
/data/logs      aoc(ro,sync,all_squash)
/data/rawimages aoc(ro,sync,all_squash)
/data/telem     aoc(ro,sync,all_squash)
HERE
    sudo exportfs -a
fi
if [[ $MAGAOX_ROLE == AOC ]]; then
    for remote in rtc icc; do
        for path in /data/logs /data/rawimages /data/telem; do
            if ! grep -q "$remote:$path"; then
                mountpoint="/data/$role$path"
                mkdir -p $mountpoint
                echo "$remote:$path 	nfs	defaults	0 0" | sudo tee -a /etc/fstab
                sudo mount $mountpoint || true
            fi
        done
    done
done
