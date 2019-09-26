#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

source /etc/os-release
if [[ $ID == ubuntu ]]; then
    sudo apt install -y \
        qt5-default \
        libqwt-qt5-dev \
        x11-apps \
    ;
elif [[ $ID == centos && $VERSION_ID == 7 ]]; then
    sudo yum install -y \
        qt5-qtbase-devel \
        xorg-x11-apps \
    ;
    sudo bash -l "$DIR/install_qwt.sh"
fi
