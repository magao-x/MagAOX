#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

log_info "Make Extra Packages for Enterprise Linux available in /etc/yum.repos.d/"
if ! dnf config-manager -h >/dev/null; then
    dnf install -y 'dnf-command(config-manager)' || exit 1
fi
dnf config-manager --set-enabled crb || exit 1
dnf install -y epel-release || exit 1

# needed for (at least) git:
yum groupinstall -y 'Development Tools' || exit 1

# Search /usr/local/lib by default for dynamic library loading
echo "/usr/local/lib" | tee /etc/ld.so.conf.d/local.conf || exit 1
ldconfig -v || exit 1

# Install build tools and utilities
yum install -y \
    util-linux-user \
    kernel-devel \
    kernel-modules-extra \
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
    rsync \
    lapack-devel \
    python \
    fftw \
    fftw-devel \
    fftw-libs \
    fftw-libs-double \
    fftw-libs-single \
    fftw-libs-long \
    fftw-static \
|| exit 1

if [[ $(uname -p) == "x86_64" ]]; then
    yum install -y fftw-libs-quad || exit 1
else
    log_info "libfftw3-quad not available on $(uname -p) host"
fi


# For some reason, pkg-config doesn't automatically look here?
mkdir -p /etc/profile.d/ || exit 1
echo "export PKG_CONFIG_PATH=\${PKG_CONFIG_PATH-}:/usr/local/lib/pkgconfig" > /etc/profile.d/99-pkg-config.sh || exit 1
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

if [[ $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == AOC ]]; then
    dnf config-manager --add-repo https://pkgs.tailscale.com/stable/rhel/9/tailscale.repo || exit 1
    dnf install tailscale || exit 1
    systemctl enable --now tailscaled || exit 1
fi


# install postgresql 15 client for RHEL 9
dnf module install -y postgresql:15/client || exit 1

# set up the postgresql server
if [[ $MAGAOX_ROLE == AOC && ! -e /var/lib/pgsql ]]; then
    dnf module install -y postgresql:15/server || exit 1
    postgresql-setup --initdb || exit 1
fi
