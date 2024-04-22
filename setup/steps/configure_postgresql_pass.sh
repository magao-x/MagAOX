#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
if ! sudo stat /opt/MagAOX/secrets/xtelemdb_password &> /dev/null; then
    sudo rm -f /opt/MagAOX/secrets/xtelemdb_password
    sudo touch /opt/MagAOX/secrets/xtelemdb_password
    sudo chown xsup:magaox /opt/MagAOX/secrets/xtelemdb_password
    sudo chmod u=r,g=,o= /opt/MagAOX/secrets/xtelemdb_password
    echo 'extremeAO!' | sudo tee -a /opt/MagAOX/secrets/xtelemdb_password
    log_info "Default xtelem database password written to /opt/MagAOX/secrets. Update and synchronize with other MagAO-X instrument computers."
else
    log_info "/opt/MagAOX/secrets/xtelemdb_password exists, not modifying"
fi