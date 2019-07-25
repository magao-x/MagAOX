#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -euo pipefail
source $DIR/_common.sh
mkdir -p /opt/MagAOX/vendor
cd /opt/MagAOX/vendor
/bin/sudo $DIR/steps/install_rt_kernel_pinned.sh
if [[ "$EUID" == 0 ]]; then
    log_error "Can't add you to the magaox-dev group when you're running as root! Aborting."
    exit 1
fi
if ! grep "hardened_usercopy=off" /etc/default/grub; then
    # without this option, the ALPAO DM driver (really the Interface Corp card driver) will
    # trigger protections against suspicious copying between kernel and userspace
    # and *bring down the whole system* (by rebooting when you try to run
    # https://github.com/magao-x/ALPAO-interface/blob/master/initalpaoPCIe )
    echo GRUB_CMDLINE_LINUX_DEFAULT="hardened_usercopy=off" | sudo tee -a /etc/default/grub
fi
/bin/sudo $DIR/setup_users_and_groups.sh
log_success "Installed PREEMPT_RT Linux kernel and configured groups. Reboot before proceeding."
