#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

yum install -y \
    zlib-devel \
    libudev-devel \
    ncurses-devel \
    nmap-ncat \
    lm_sensors \
    hddtemp \
    readline-devel \
    pkg-config \
    bison \
    flex \
    dialog \
;

# For some reason, pkg-config doesn't automatically look here?
echo "export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig" > /etc/profile.d/pkg-config-path.sh
