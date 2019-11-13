#!/bin/bash
# If not started as root, sudo yourself
if [[ "$EUID" != 0 ]]; then
    sudo bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

NODEJS_VERSION=v12.13.0
if [[ ! -d /opt/node ]]; then
    mkdir -p /opt/node
    _cached_fetch  https://nodejs.org/dist/${NODEJS_VERSION}/node-${NODEJS_VERSION}-linux-x64.tar.xz node-${NODEJS_VERSION}-linux-x64.tar.xz
    tar --strip-components=1 -xvf node-${NODEJS_VERSION}-linux-x64.tar.xz -C /opt/node
fi
if [[ ! -e /etc/profile.d/node_bin_path.sh ]]; then
    echo "export PATH=\"\$PATH:/opt/node/bin\"" | sudo tee /etc/profile.d/node_bin_path.sh
fi
if [[ -z $(command -v yarn) ]]; then
    npm install --global yarn
fi