#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh
mkdir -p /opt/MagAOX/vendor
cd /opt/MagAOX/vendor
/bin/sudo $DIR/steps/install_rt_kernel_pinned.sh
if [[ "$EUID" == 0 ]]; then
    log_error "Can't add you to the magaox-dev group when you're running as root! Aborting."
    exit 1
fi
/bin/sudo $DIR/steps/setup_users_and_groups.sh
log_success "Installed PREEMPT_RT Linux kernel and configured groups. Reboot before proceeding."
