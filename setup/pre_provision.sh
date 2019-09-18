#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -euo pipefail
source $DIR/_common.sh
if [[ "$EUID" == 0 ]]; then
    log_error "Can't add you to the magaox-dev group when you're running as root! Aborting."
    exit 1
fi

source /etc/os-release
# without hardened_usercopy=off, the ALPAO DM driver (really the Interface Corp card driver) will
# trigger protections against suspicious copying between kernel and userspace
# and *bring down the whole system* (by rebooting when you try to run
# https://github.com/magao-x/ALPAO-interface/blob/master/initalpaoPCIe )
ALPAO_CMDLINE_FIX="hardened_usercopy=off"
# make the PCIe expansion card work
PCIEXPANSION_CMDLINE_FIX="pci=noaer"
# disable the slow Spectre mitigations
SPECTRE_CMDLINE_FIX="noibrs noibpb nopti nospectre_v2 nospectre_v1 l1tf=off nospec_store_bypass_disable no_stf_barrier mds=off mitigations=off"
# Put it all together
DESIRED_CMDLINE="$ALPAO_CMDLINE_FIX $PCIEXPANSION_CMDLINE_FIX $SPECTRE_CMDLINE_FIX"
if [[ $ID == ubuntu ]]; then
    log_info "Skipping RT kernel install on Ubuntu"
    install_rt=false
else
    if ! grep "$DESIRED_CMDLINE" /etc/default/grub; then
        echo GRUB_CMDLINE_LINUX_DEFAULT=\""$DESIRED_CMDLINE"\" | sudo tee -a /etc/default/grub
        sudo grub2-mkconfig -o /boot/grub2/grub.cfg
        log_success "Applied kernel command line tweaks for ALPAO, Spectre, PCIe expansion"
    fi
    if [[ $(uname -v) != *"PREEMPT RT"* ]]; then
        sudo mkdir -p /opt/MagAOX/vendor
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
