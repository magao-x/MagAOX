#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

if [[ $MAGAOX_ROLE == vm ]]; then
    sudo systemctl disable firewalld || true
    sudo systemctl stop firewalld || true
    log_info "Stopped firewall because this is a CentOS 7 VM"
fi

echo "export PATH=\"\$PATH:/usr/local/bin\"" | sudo tee /etc/profile.d/usr-local-bin-on-path.sh
log_success "Added /usr/local/bin to PATH via /etc/profile.d/usr-local-bin-on-path.sh"