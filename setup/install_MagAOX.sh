#!/bin/bash
source /opt/rh/devtoolset-7/enable
export PATH="/usr/local/bin:$PATH"
set -exuo pipefail
IFS=$'\n\t'

if [[ ! -e /opt/MagAOX/config ]]; then
    echo "Cloning new copy of MagAOX config files"
    /bin/sudo git clone https://github.com/magao-x/config.git /opt/MagAOX/config
    cd /opt/MagAOX/config
    echo "Config branches available:"
    git branch
    chown -R :magaox-dev /opt/MagAOX/config
    chmod -R g=rwX /opt/MagAOX/config
    # n.b. can't be recursive because g+s on files means something else
    # so we find all directories and individually chmod them:
    find /opt/MagAOX/config -type d -exec chmod -v g+s {} \;
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