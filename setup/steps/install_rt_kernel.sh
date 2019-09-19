#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

sudo yum install -y yum-versionlock

sudo tee /etc/yum.repos.d/CentOS-rt.repo >/dev/null <<EOF
# CentOS-rt.repo

[rt]
name=CentOS-7 - rt
baseurl=http://mirror.centos.org/centos/\$releasever/rt/\$basearch/
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-7
EOF

sudo yum update -y || true
if ! rpm -q tuned-2.9.0-1.el7fdp.noarch; then
    # Newer version installed by default, too new for the tuned-profiles-realtime package:
    sudo yum remove -y tuned
fi
sudo yum install -y tuned-2.9.0-1.el7fdp.noarch
sudo yum versionlock tuned
sudo yum install -y kernel-rt-3.10.0-957.21.3.rt56.935.el7 kernel-rt-devel-3.10.0-957.21.3.rt56.935.el7
sudo yum versionlock kernel-rt
sudo yum versionlock kernel
# Dependencies Resolved

# =====================================================================================================
#  Package                        Arch          Version                              Repository   Size
# =====================================================================================================
# Installing:
#  kernel-rt                      x86_64        3.10.0-957.21.3.rt56.935.el7         rt           45 M
# Installing for dependencies:
#  python-ethtool                 x86_64        0.8-8.el7                            base         34 k
#  rt-setup                       x86_64        2.0-6.el7                            rt           20 k
#  tuna                           noarch        0.13-9.el7                           base        141 k
#  tuned-profiles-realtime        noarch        2.9.0-1.el7fdp                       rt           26 k

# Transaction Summary
# =====================================================================================================
# Install  1 Package (+4 Dependent packages)

# Total download size: 45 M
# Installed size: 169 M
# Is this ok [y/d/N]:
