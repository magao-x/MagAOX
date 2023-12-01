#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
apt-get update

NEEDRESTART_SUSPEND=yes apt install -y \
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
    netcat \
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
    libopenblas-openmp-dev \
    liblapack-dev \
    liblapacke-dev \
;

if [[ $MAGAOX_ROLE == vm ]]; then
    NEEDRESTART_SUSPEND=yes apt install -y xauth
fi
if [[ $(uname -p) == "aarch64" ]]; then
    # since we won't install MKL:
    NEEDRESTART_SUSPEND=yes apt install -y libopenblas-openmp-dev liblapack-dev liblapacke-dev
fi
