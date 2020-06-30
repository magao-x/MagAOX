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
        mesa-libGL-devel \
    ;
fi

if [[ $MAGAOX_ROLE == vm ]]; then
    # For some reason, Qt won't hear any keyboard events unless this is set.
    # (Hinted at by: "Qt: Failed to create XKB context!")
    echo "export QT_XKB_CONFIG_ROOT=/usr/lib/kbd/keymaps/xkb" | sudo tee /etc/profile.d/qt_xkb_config_env_var.sh
fi
