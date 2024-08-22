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
        read -p "Role: " roleinput
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
set -uo pipefail

source /etc/os-release
# without hardened_usercopy=off, the ALPAO DM driver (really the Interface Corp card driver) will
# trigger protections against suspicious copying between kernel and userspace
# and *bring down the whole system* (by rebooting when you try to run
# https://github.com/magao-x/ALPAO-interface/blob/master/initalpaoPCIe )
ALPAO_CMDLINE_FIX="hardened_usercopy=off"
# Try to work around iommu issue with RTC2
IOMMU_FIX="iommu=pt"
# make the PCIe expansion card work
PCIEXPANSION_CMDLINE_FIX="pci=noaer"
# disable the slow Spectre mitigations
SPECTRE_CMDLINE_FIX="noibrs noibpb nopti nospectre_v2 nospectre_v1 l1tf=off nospec_store_bypass_disable no_stf_barrier mds=off mitigations=off"
# disable 3rd party nvidia drivers
NVIDIA_DRIVER_FIX="rd.driver.blacklist=nouveau nouveau.modeset=0"

# Put it all together
if [[ $MAGAOX_ROLE == "workstation" ]]; then
    DESIRED_CMDLINE="nosplash $NVIDIA_DRIVER_FIX"
else
    DESIRED_CMDLINE="nosplash $NVIDIA_DRIVER_FIX $ALPAO_CMDLINE_FIX $PCIEXPANSION_CMDLINE_FIX $SPECTRE_CMDLINE_FIX $IOMMU_FIX"
fi

if ! sudo grep -r "$DESIRED_CMDLINE" /boot/loader/entries; then
    sudo cp /etc/default/grub /etc/default/grub.bak || exit 1
    sudo grubby --update-kernel=ALL --args="$DESIRED_CMDLINE" || exit 1

    if [[ -d /boot/grub2 ]]; then
        sudo grub2-mkconfig -o /boot/grub2/grub.cfg || exit 1
    elif [[ -d /boot/grub ]]; then
        sudo update-grub || exit 1
    else
        exit_with_error "Where's grub gotten to?"
    fi
    log_success "Applied kernel command line tweaks"
fi

if [[ -d /etc/initramfs-tools ]]; then
    log_info "Disabling hibernate/resume support in initramfs"
    if ! grep 'RESUME=none' /etc/initramfs-tools/conf.d/resume; then
        echo "RESUME=none" | sudo tee /etc/initramfs-tools/conf.d/resume || exit 1
        sudo update-initramfs -u -k all || exit 1
    fi
fi

if [[ ! -e /etc/modprobe.d/blacklist-nouveau.conf ]]; then
    echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null || exit 1
    echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf > /dev/null || exit 1
    if [[ $ID == ubuntu ]]; then
        sudo update-initramfs -u || exit 1
    fi
    log_success "Blacklisted nouveau nvidia driver"
else
    log_info "nouveau nvidia driver blacklist entry exists"
fi

if ! grep "Storage=persistent" /etc/systemd/journald.conf; then
    echo "Storage=persistent" | sudo tee -a /etc/systemd/journald.conf || exit 1
    log_success "Enabled persistent systemd journald log across reboots"
fi

$DIR/setup_users_and_groups.sh || exit_with_error "Failed to set up users and groups"
log_success "Created users and configured groups"
log_info "You will need to use grub2-install with the right devices to get RAID1 reliability for boot"
log_success "Reboot before proceeding"
