#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == workstation ]]; then
    if [[ $(systemctl get-default) != "graphical.target" ]]; then 
        sudo yum groupinstall "KDE Plasma Workspaces" -y
        log_success "KDE Plasma Workspaces installed"
        sudo systemctl set-default graphical.target
        log_success "Setting 'graphical.target' as systemd default, reboot or 'systemctl isolate' to switch"
    else
        log_info "Already have 'graphical.target' set as systemd default"
    fi
else
    log_info "Skipping graphical environment on $MAGAOX_ROLE"
fi
