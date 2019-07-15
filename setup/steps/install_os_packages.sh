#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
# needed for (at least) git:
yum groupinstall -y 'Development Tools'

# Install EPEL
# use || true so it's not an error if already installed:
yum install -y http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm || true

# changes the set of available packages, making devtoolset-7 available
yum -y install centos-release-scl

# Install build tools and utilities
yum install -y \
    cmake \
    cmake3 \
    vim \
    nano \
    wget \
    htop
