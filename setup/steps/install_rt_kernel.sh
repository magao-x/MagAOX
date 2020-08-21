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
# Newer version installed by default, but the tuned-profiles-realtime package lags, so reinstall:
sudo yum remove -y tuned
sudo yum install -y tuned-profiles-realtime
sudo yum versionlock tuned
# Install latest kernel and pin it if we're installing the first time:
sudo yum install -y kernel-rt kernel-rt-devel
sudo yum versionlock kernel-rt
sudo yum versionlock kernel

# Patch kernel module makefile for CONFIG_RETPOLINE
# https://bitsanddragons.wordpress.com/2018/03/20/config_retpoliney-error-on-centos-7/
sed \
    -i 's/$(error CONFIG_RETPOLINE=y/$(echo CONFIG_REPOLINE=y/' \
    /usr/src/kernels/$(uname -r)/arch/x86/Makefile
