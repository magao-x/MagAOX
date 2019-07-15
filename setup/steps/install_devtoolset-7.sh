#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
# install and enable devtoolset-7 for all users
# Note: this only works on interactive shells! There is a bug in SCL
# that breaks sudo argument parsing when SCL is enabled
# (https://bugzilla.redhat.com/show_bug.cgi?id=1319936)
# so we don't want it enabled when, e.g., Vagrant
# sshes in to change things. (Complete sudo functionality
# is available to interactive shells by specifying /bin/sudo.)
yum -y install devtoolset-7
echo "if tty -s; then source /opt/rh/devtoolset-7/enable; fi" | tee /etc/profile.d/devtoolset-7.sh
set +u
source /opt/rh/devtoolset-7/enable
set -u
# Search /usr/local/lib by default for dynamic library loading
echo "/usr/local/lib" | tee /etc/ld.so.conf.d/local.conf
ldconfig -v

