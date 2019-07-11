#!/bin/bash
tee /etc/yum.repos.d/CentOS-rt.repo >/dev/null <<EOF
# CentOS-rt.repo

[rt]
name=CentOS-7 - rt
baseurl=http://mirror.centos.org/centos/\$releasever/rt/\$basearch/
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-7
EOF
yum update -y
# Installed by default, but too new for the tuned-profiles-realtime package:
yum remove -y tuned-2.10.0-6.el7_6.3.noarch || true
# Provided by the (rt) repo:
yum install -y tuned-2.9.0-1.el7fdp.noarch || true
yum install -y kernel-rt kernel-rt-devel rt-tests tuned-profiles-realtime
