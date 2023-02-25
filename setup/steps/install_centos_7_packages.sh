#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -euo pipefail

# needed for (at least) git:
yum groupinstall -y 'Development Tools'

# Install EPEL
# use || true so it's not an error if already installed:
yum install -y http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm || true

# Install repo with recent tmux
# recommended by https://github.com/tmux/tmux/wiki/Installing#red-hat-enterprise-linux--centos-rpms
yum install -y http://galaxy4.net/repo/galaxy4-release-7-current.noarch.rpm || true

# changes the set of available packages, making devtoolset-7 available
yum -y install centos-release-scl  || exit_error "Failed to enable devtoolsets"
# install and enable devtoolset-7 for all users
# Note: this only works on interactive shells! There is a bug in SCL
# that breaks sudo argument parsing when SCL is enabled
# (https://bugzilla.redhat.com/show_bug.cgi?id=1319936)
# so we don't want it enabled when, e.g., Vagrant
# sshes in to change things. (Complete sudo functionality
# is available to interactive shells by specifying /bin/sudo.)
yum -y install devtoolset-7
cat << 'EOF' | tee /etc/profile.d/devtoolset-7.sh
if tty -s; then
  if [[ $USER != root ]]; then source /opt/rh/devtoolset-7/enable; fi
fi
EOF
set +u
source /opt/rh/devtoolset-7/enable
set -u
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
    ntp \
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
    log4cxx-devel \
    chrony \
    gdb \
    yum-utils \
    yum-versionlock \
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
;

if [[ $MAGAOX_ROLE == vm ]]; then
    yum install -y xorg-x11-xauth
fi

alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake3 20 \
    --slave /usr/local/bin/ctest ctest /usr/bin/ctest3 \
    --slave /usr/local/bin/cpack cpack /usr/bin/cpack3 \
    --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake3 \
    --family cmake

alternatives --install /usr/local/bin/qmake qmake /usr/bin/qmake-qt5 20

# Undo installing stable ZeroMQ without draft APIs
yum remove -y zeromq-devel libzmq5 || true
yum-config-manager --disable network_messaging_zeromq_release-stable || true
# Install stable ZeroMQ with draft APIs
yum-config-manager --add-repo https://download.opensuse.org/repositories/network:/messaging:/zeromq:/release-draft/CentOS_7/network:messaging:zeromq:release-draft.repo
yum install -y zeromq-devel libzmq5

# For some reason, pkg-config doesn't automatically look here?
echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig" > /etc/profile.d/pkg-config-path.sh

mkdir -p /opt/MagAOX/vendor
cd /opt/MagAOX/vendor
rpmFile=cpuset-1.6-lp154.59.1.noarch.rpm

# get _cached_fetch
source $DIR/../_common.sh

_cached_fetch https://download.opensuse.org/repositories/hardware/15.4/noarch/$rpmFile $rpmFile
sudo yum install -y $rpmFile || true
cat <<'HERE' | sudo tee /etc/cset.conf || exit 1
mountpoint = /sys/fs/cgroup/cpuset
HERE