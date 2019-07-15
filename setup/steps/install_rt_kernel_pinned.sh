#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -e

function getandinstall() {
    baseurl=$1
    name=$2
    version=$3
    filename=$name-$version.rpm
    log_info "Downloading $baseurl/$filename"
    if [[ ! -e $filename ]]; then
        curl -OL $baseurl/$filename
        log_success "Downloaded $filename from $baseurl"
    else
        log_info "$filename RPM present"
    fi
    rpm -i $filename || true  # no error if already installed
    yum install -y $name-$version
}

CENTOS_BASE=http://mirror.centos.org/centos/7/os/x86_64/Packages
CENTOS_UPDATES=http://mirror.centos.org/centos/7/updates/x86_64/Packages
CENTOS_RT=http://mirror.centos.org/centos/7/rt/x86_64/Packages

# Installed by default, but too new for the tuned-profiles-realtime package:
yum remove -y tuned

getandinstall $CENTOS_BASE    libnl                   1.1.4-3.el7.x86_64
getandinstall $CENTOS_BASE    python-ethtool          0.8-7.el7.x86_64
getandinstall $CENTOS_BASE    tuna                    0.13-6.el7.noarch
getandinstall $CENTOS_BASE    snappy                  1.1.0-3.el7.x86_64
getandinstall $CENTOS_BASE    dracut-network          033-554.el7.x86_64
getandinstall $CENTOS_UPDATES kexec-tools             2.0.15-21.el7_6.3.x86_64
getandinstall $CENTOS_RT      tuned                   2.9.0-1.el7fdp.noarch
getandinstall $CENTOS_RT      tuned-profiles-realtime 2.9.0-1.el7fdp.noarch
getandinstall $CENTOS_RT      rt-setup                2.0-6.el7.x86_64
getandinstall $CENTOS_RT      kernel-rt-devel         3.10.0-957.21.3.rt56.935.el7.x86_64
getandinstall $CENTOS_RT      kernel-rt               3.10.0-957.21.3.rt56.935.el7.x86_64
