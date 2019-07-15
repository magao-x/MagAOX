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
    hddtemp
