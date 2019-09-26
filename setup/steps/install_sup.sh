#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/source
if [[ $MAGAOX_ROLE == vm ]]; then
    parentdir=/vagrant/vm
else
    parentdir=/opt/MagAOX/source
fi
clone_or_update_and_cd magao-x sup $parentdir
pip install -e .
UNIT_PATH=/etc/systemd/system/
if [[ ! -e $UNIT_PATH/sup.service ]]; then
    cp /opt/MagAOX/config/sup.service $UNIT_PATH/sup.service
fi
sudo systemctl enable sup.service || true
sudo systemctl restart sup.service || true
