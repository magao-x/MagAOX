#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -euo pipefail

# needed for (at least) git:
yum groupinstall -y 'Development Tools'

# Search /usr/local/lib by default for dynamic library loading
echo "/usr/local/lib" | tee /etc/ld.so.conf.d/local.conf
ldconfig -v

# Install build tools and utilities
yum install -y \
    which \
    openssh \
    cmake3 \
    vim \
    nano \
    wget \
    mlocate \
    htop \
    zlib-devel \
    libudev-devel \
    ncurses-devel \
    nmap-ncat \
    lm_sensors \
    hddtemp \
    readline-devel \
    pkgconfig \
    bison \
    flex \
    dialog \
    autossh \
    check-devel \
    subunit-devel \
    pciutils \
    libusb-devel \
    libusbx-devel \
    usbutils \
    tmux \
    boost-devel \
    gsl \
    gsl-devel \
    bc \
    chrony \
    gdb \
    yum-utils \
    ntfs-3g \
    screen \
    which \
    sudo \
    sysstat \
    fuse \
    psmisc \
    podman \
    nethogs \
    shadow-utils \
    nfs-utils \
    pybind11-devel \
;

if [[ $MAGAOX_ROLE == vm ]]; then
    yum install -y xorg-x11-xauth
fi

# alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake3 20 \
#     --slave /usr/local/bin/ctest ctest /usr/bin/ctest3 \
#     --slave /usr/local/bin/cpack cpack /usr/bin/cpack3 \
#     --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake3 \
#     --family cmake

# alternatives --install /usr/local/bin/qmake qmake /usr/bin/qmake-qt5 20

# # Undo installing stable ZeroMQ without draft APIs
# yum remove -y zeromq-devel libzmq5 || true
# yum-config-manager --disable network_messaging_zeromq_release-stable || true

# # For some reason, pkg-config doesn't automatically look here?
mkdir -p /etc/profile.d/
echo "export PKG_CONFIG_PATH=\${PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig" > /etc/profile.d/99-pkg-config.sh
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
# mkdir -p /opt/MagAOX/vendor
# cd /opt/MagAOX/vendor
# rpmFile=cpuset-1.6-lp154.59.1.noarch.rpm

# # get _cached_fetch
# source $DIR/../_common.sh

# _cached_fetch https://download.opensuse.org/repositories/hardware/15.4/noarch/$rpmFile $rpmFile
# sudo yum install -y $rpmFile || true
# cat <<'HERE' | sudo tee /etc/cset.conf || exit 1
# mountpoint = /sys/fs/cgroup/cpuset
# HERE

if [[ $(uname -p) == "aarch64" ]]; then
    # since we won't install MKL:
    yum install -y openblas-devel lapack-devel
fi