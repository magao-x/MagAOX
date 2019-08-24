#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
apt-get update

apt install -y \
    ssh \
    build-essential \
    gfortran \
    udev \
    rsync \
    git \
    cmake \
    curl \
    cpio \
    vim \
    nano \
    htop \
    strace \
    zlib1g-dev \
    libudev-dev \
    libncurses5-dev \
    netcat \
    lm-sensors \
    hddtemp \
    libreadline-dev \
    pkg-config \
    bison \
    flex \
    dialog \
    autossh \
    check \
    libsubunit-dev \
    pciutils \
    libusb-1.0-0-dev \
    usbutils \
    tmux \
    libboost-all-dev \
    libgsl-dev \
    bc \
    libczmq-dev \
    liblog4cxx-dev \
;
