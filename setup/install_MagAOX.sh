#!/bin/bash
source /opt/rh/devtoolset-7/enable
export PATH="/usr/local/bin:$PATH"
set -exuo pipefail
IFS=$'\n\t'

if [[ ! -e /opt/MagAOX/config ]]; then
    echo "Cloning new copy of MagAOX config files"
    git clone https://github.com/magao-x/config.git /opt/MagAOX/config
    cd /opt/MagAOX/config
    echo "Config branches available:"
    git branch
    chmod -R g+s /opt/MagAOX/config
fi
echo "Building flatlogs"
cd /opt/MagAOX/source/MagAOX/flatlogs/src
make
make install
echo "Building MagAOX"
cd /opt/MagAOX/source/MagAOX
make setup
make all
make install