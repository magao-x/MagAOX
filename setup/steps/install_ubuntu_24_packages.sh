#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
apt-get update

apt-get install -y \
    sudo \
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
    libncursesw5-dev \
    netcat-traditional \
    lm-sensors \
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
    liblog4cxx-dev \
    chrony \
    gdb \
    unzip \
    cpuset \
    nfs-common \
    nfs-kernel-server \
    tree \
    linux-headers-generic \
    liblapack-dev \
    liblapacke-dev \
    podman \
;

if [[ $VM_KIND != "none" ]]; then
    apt install -y xauth
fi
