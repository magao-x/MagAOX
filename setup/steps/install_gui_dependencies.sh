#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

source /etc/os-release
if [[ $ID == ubuntu ]]; then
    sudo apt install -y \
        x11-apps \
    ;
elif [[ $ID == centos && $VERSION_ID == 7 ]]; then
    sudo yum install -y \
        xorg-x11-apps \
        kate \
        wmctrl \
    ;
fi
