#!/bin/bash
# If not started as root, sudo yourself
if [[ "$EUID" != 0 ]]; then
    sudo bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
# See https://docs.docker.com/install/linux/docker-ce/centos/#install-from-a-package
cd /opt/MagAOX/vendor
BASEURL="https://download.docker.com/linux/centos/7/x86_64/stable/Packages"
DOCKER_RPMFILE="docker-ce-19.03.4-3.el7.x86_64.rpm"
CONTAINERD_RPMFILE="containerd.io-1.2.6-3.3.el7.x86_64.rpm"
DOCKERCLI_RPMFILE="docker-ce-cli-19.03.4-3.el7.x86_64.rpm"

_cached_fetch $BASEURL/$CONTAINERD_RPMFILE $CONTAINERD_RPMFILE
_cached_fetch $BASEURL/$DOCKERCLI_RPMFILE $DOCKERCLI_RPMFILE
_cached_fetch $BASEURL/$DOCKER_RPMFILE $DOCKER_RPMFILE

yum install -y $DOCKER_RPMFILE $DOCKERCLI_RPMFILE $CONTAINERD_RPMFILE || yum upgrade -y $DOCKER_RPMFILE $DOCKERCLI_RPMFILE $CONTAINERD_RPMFILE
systemctl start docker
docker run hello-world