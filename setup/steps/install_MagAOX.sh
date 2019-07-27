#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

function clone_or_update_and_cd() {
    if [[ ! -e /opt/MagAOX/$1 ]]; then
        echo "Cloning new copy of MagAOX config files"
        git clone https://github.com/magao-x/$1.git /tmp/$1
        sudo rsync -av /tmp/$1/ /opt/MagAOX/$1/
        cd /opt/MagAOX/$1
        log_success "Cloned new /opt/MagAOX/$1"
    else
        cd /opt/MagAOX/$1
        git pull
        log_success "Updated /opt/MagAOX/$1"
    fi
    sudo chown -R :magaox-dev /opt/MagAOX/$1
    sudo chmod -R g=rwX /opt/MagAOX/$1
    # n.b. can't be recursive because g+s on files means something else
    # so we find all directories and individually chmod them:
    sudo find /opt/MagAOX/$1 -type d -exec chmod g+s {} \;
    log_success "Normalized permissions on /opt/MagAOX/$1"
}

clone_or_update_and_cd config
echo "Config branches available:"
git branch

clone_or_update_and_cd calib

echo "Building flatlogs"
cd /opt/MagAOX/source/MagAOX/flatlogs/src
make
make install
echo "Building MagAOX"
cd /opt/MagAOX/source/MagAOX
make setup
make all
make install
