#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
apt-get update || exit 1

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
    libfftw3-bin \
    libfftw3-dev \
    libfftw3-doc \
    libfftw3-single \
    libfftw3-long \
    libfftw3-double \
|| exit 1

if [[ $(uname -p) == "x86_64" ]]; then
    apt-get install -y libfftw3-quad || exit 1
else
    log_info "libfftw3-quad not available on $(uname -p) host"
fi
