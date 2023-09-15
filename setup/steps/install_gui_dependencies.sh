#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

source /etc/os-release
if [[ $ID == ubuntu ]]; then
    sudo -i apt install -y \
        x11-apps \
        libgl-dev \
        qtbase5-dev \
        qtchooser \
        qt5-qmake \
        qtbase5-dev-tools \
        libqt5svg5-dev \
        wmctrl \
        libqwt-qt5-dev/jammy \
    ;
elif [[ $ID == centos && $VERSION_ID == 7 ]]; then
    sudo yum install -y \
        xorg-x11-apps \
        kate \
        wmctrl \
        mesa-libGL-devel \
    ;
elif [[ $ID == rocky && $VERSION_ID == "9."* ]]; then
    sudo dnf install -y \
        qt5-devel \
        qwt-qt5-devel \
    ;
fi

if [[ $MAGAOX_ROLE == vm ]]; then
    # For some reason, Qt won't hear any keyboard events unless this is set.
    # (Hinted at by: "Qt: Failed to create XKB context!")
    echo "export QT_XKB_CONFIG_ROOT=/usr/lib/kbd/keymaps/xkb" | sudo tee /etc/profile.d/qt_xkb_config_env_var.sh
fi
