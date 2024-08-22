#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

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
        xauth \
    || exit 1

    if [[ $VERSION_ID = "24.04" ]]; then
        sudo -i apt install -y libqwtmathml-qt5-dev || exit 1
    else
        sudo -i apt install -y libqwt-qt5-dev/jammy || exit 1
    fi
elif [[ $ID == rocky && $VERSION_ID == "9."* ]]; then
    sudo dnf install -y \
        qt5-devel \
        qwt-qt5-devel \
        wmctrl \
        xorg-x11-xauth \
    || exit 1
fi

# For some reason, Qt won't hear any keyboard events unless this is set.
# (Hinted at by: "Qt: Failed to create XKB context!")
echo "export QT_XKB_CONFIG_ROOT=/usr/lib/kbd/keymaps/xkb" | sudo tee /etc/profile.d/qt_xkb_config_env_var.sh || exit 1
