#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

log_info "Make Extra Packages for Enterprise Linux available in /etc/yum.repos.d/"
dnf config-manager --set-enabled crb
dnf install -y epel-release

# needed for (at least) git:
yum groupinstall -y 'Development Tools'

# Search /usr/local/lib by default for dynamic library loading
echo "/usr/local/lib" | tee /etc/ld.so.conf.d/local.conf
ldconfig -v

# Install build tools and utilities
yum install -y \
    kernel-devel \
    gcc-gfortran \
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

# For some reason, pkg-config doesn't automatically look here?
mkdir -p /etc/profile.d/
echo "export PKG_CONFIG_PATH=\${PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig" > /etc/profile.d/99-pkg-config.sh
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

if [[ $(uname -p) == "aarch64" ]]; then
    # since we won't install MKL:
    yum install -y openblas-devel lapack-devel
fi

if [[ $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == AOC ]]; then
    dnf config-manager --add-repo https://pkgs.tailscale.com/stable/rhel/9/tailscale.repo
    dnf install tailscale
    systemctl enable --now tailscaled
    tailscale up
fi