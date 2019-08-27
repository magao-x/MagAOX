#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -euo pipefail
source $DIR/_common.sh
if [[ "$EUID" == 0 ]]; then
    log_error "Can't add you to the magaox-dev group when you're running as root! Aborting."
    exit 1
fi

source /etc/os-release
if [[ $ID == ubuntu ]]; then
    log_info "Skipping RT kernel install on Ubuntu"
    install_rt=false
else
    if ! grep "hardened_usercopy=off" /etc/default/grub; then
        # without this option, the ALPAO DM driver (really the Interface Corp card driver) will
        # trigger protections against suspicious copying between kernel and userspace
        # and *bring down the whole system* (by rebooting when you try to run
        # https://github.com/magao-x/ALPAO-interface/blob/master/initalpaoPCIe )
        echo GRUB_CMDLINE_LINUX_DEFAULT="hardened_usercopy=off" | sudo tee -a /etc/default/grub
        sudo grub2-mkconfig -o /boot/grub2/grub.cfg
        log_success "Applied kernel command line tweak for ALPAO"
    fi
    if [[ $(uname -v) != *"PREEMPT RT"* ]]; then
        mkdir -p /opt/MagAOX/vendor
        cd /opt/MagAOX/vendor
        sudo $DIR/steps/install_rt_kernel_pinned.sh
        log_success "Installed PREEMPT_RT Linux kernel packages"
    fi
    install_rt=true
fi
sudo $DIR/setup_users_and_groups.sh
log_success "Created users and configured groups"
if [[ $install_rt == true ]]; then
    sudo $DIR/steps/install_rt_kernel_pinned.sh
    log_success "Reboot before proceeding"
else
    log_success "Log out and back in before proceeding"
fi
