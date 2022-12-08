#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $DIR/_common.sh
if [[ "$EUID" == 0 ]]; then
    log_error "Can't add you to the magaox-dev group when you're running as root! Aborting."
    exit 1
fi

if [[ -z $MAGAOX_ROLE ]]; then
    MAGAOX_ROLE=""
    echo "Choose the role for this machine"
    echo "    AOC - Adaptive optics Operator Computer"
    echo "    RTC - Real Time control Computer"
    echo "    ICC - Instrument Control Computer"
    echo "    TIC - Testbed Instrument Computer"
    echo "    TOC - Testbed Operator Computer"
    echo "    workstation - Any other MagAO-X workstation"
    echo
    while [[ -z $MAGAOX_ROLE ]]; do
        read -p "Role:" roleinput
        case $roleinput in
            AOC)
                MAGAOX_ROLE=AOC
                ;;
            RTC)
                MAGAOX_ROLE=RTC
                ;;
            ICC)
                MAGAOX_ROLE=ICC
                ;;
            TIC)
                MAGAOX_ROLE=TIC
                ;;
            TOC)
                MAGAOX_ROLE=TOC
                ;;
            workstation)
                MAGAOX_ROLE=workstation
                ;;
            *)
                echo "Must be one of AOC, RTC, ICC, TIC, TOC, or workstation."
                continue
        esac
    done
else
    echo "Already have MAGAOX_ROLE=$MAGAOX_ROLE, not prompting for it. (Edit /etc/profile.d/magaox_role.sh if it's wrong)"
fi

echo "export MAGAOX_ROLE=$MAGAOX_ROLE" | sudo tee /etc/profile.d/magaox_role.sh
export MAGAOX_ROLE
set -euo pipefail

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
# disable 3rd party nvidia drivers
NVIDIA_DRIVER_FIX="rd.driver.blacklist=nouveau nouveau.modeset=0"

# Put it all together
DESIRED_CMDLINE="nosplash $NVIDIA_DRIVER_FIX $ALPAO_CMDLINE_FIX $PCIEXPANSION_CMDLINE_FIX $SPECTRE_CMDLINE_FIX"

if ! grep "$DESIRED_CMDLINE" /etc/default/grub; then
    echo GRUB_CMDLINE_LINUX_DEFAULT=\""$DESIRED_CMDLINE"\" | sudo tee -a /etc/default/grub
    sudo grub2-mkconfig -o /boot/grub2/grub.cfg
    log_success "Applied kernel command line tweaks"
fi

if [[ ! -e /etc/modprobe.d/blacklist-nouveau.conf ]]; then
    echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
    echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
    log_success "Blacklisted nouveau nvidia driver"
else
    log_info "nouveau nvidia driver blacklist entry exists"
fi

if ! grep "Storage=persistent" /etc/systemd/journald.conf; then
    echo "Storage=persistent" | sudo tee -a /etc/systemd/journald.conf
    log_success "Enabled persistent systemd journald log across reboots"
fi

$DIR/setup_users_and_groups.sh
log_success "Created users and configured groups"

log_success "Reboot before proceeding"
