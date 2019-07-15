#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
tee /etc/yum.repos.d/CentOS-rt.repo >/dev/null <<EOF
# CentOS-rt.repo

[rt]
name=CentOS-7 - rt
baseurl=http://mirror.centos.org/centos/\$releasever/rt/\$basearch/
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-7
EOF
# Installed by default, but too new for the tuned-profiles-realtime package:
yum remove -y tuned-2.10.0-6.el7_6.3.noarch
# Provided by the (rt) repo:
yum install  tuned-2.9.0-1.el7fdp.noarch
yum install  kernel-rt kernel-rt-devel tuned-profiles-realtime
