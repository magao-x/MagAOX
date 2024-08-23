#!/bin/bash
set -o pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh

# CentOS + devtoolset-7 aliases sudo, but breaks command line arguments for it,
# so if we need those, we must use $_REAL_SUDO.
if [[ -e /usr/bin/sudo ]]; then
  _REAL_SUDO=/usr/bin/sudo
elif [[ -e /bin/sudo ]]; then
  _REAL_SUDO=/bin/sudo
else
  if [[ -z $(command -v sudo) ]]; then
    echo "Install sudo before provisioning"
  else
    _REAL_SUDO=$(which sudo)
  fi
fi

# Function to refresh sudo timer
refresh_sudo_timer() {
    while true; do
        $_REAL_SUDO -v
        sleep 60
    done
}

# Clear cached credentials for sudo, if they exist
sudo -K

# Start refreshing sudo timer in the background
if [[ "$(sudo -H -n true 2>&1)" ]]; then
    $_REAL_SUDO -v
    refresh_sudo_timer &
fi

# Defines $ID and $VERSION_ID so we can detect which distribution we're on
source /etc/os-release

roleScript=/etc/profile.d/magaox_role.sh
VM_KIND=$(systemd-detect-virt)
if [[ $VM_KIND != "none" ]]; then
    echo "Detected virtualization: $VM_KIND"
fi
if [[ ! -e $roleScript && ! -z $MAGAOX_ROLE ]]; then
    echo "export MAGAOX_ROLE=$MAGAOX_ROLE" | $_REAL_SUDO tee $roleScript
fi
if [[ ! -e $roleScript ]]; then
    echo "Export \$MAGAOX_ROLE in $roleScript first"
    exit 1
fi
source $roleScript
echo "Got MAGAOX_ROLE=$MAGAOX_ROLE"
export MAGAOX_ROLE

# The VM and CI provisioning doesn't run setup_users_and_groups.sh
# separately as in the instrument instructions; we have to run it
if [[ $MAGAOX_ROLE == workstation || $MAGAOX_ROLE == ci ]]; then
    bash -l "$DIR/setup_users_and_groups.sh"
fi
## Set up file structure and permissions
sudo -H bash -l "$DIR/steps/ensure_dirs_and_perms.sh" $MAGAOX_ROLE

# Install OS-packaged and a few self-built dependencies.
if [[ ! $_skip3rdPartyDeps ]]; then
    # For staged VM builds we don't want to redo the 3rd party deps
    # (even if they're mostly already done). Setting $_skip3rdPartyDeps
    # lets us skip this line:
    sudo -H bash -l "$DIR/install_third_party_deps.sh" || exit_with_error "Failed to install third-party dependencies"
fi

# Apply configuration tweaks
log_info "Applying configuration tweaks for OS and services"

if [[ $VM_KIND != "none" ]]; then
    git config --global --replace-all safe.directory '*'
    sudo -H git config --global --replace-all safe.directory '*'
fi

bash -l "$DIR/steps/configure_trusted_sudoers.sh" || exit_with_error "Could not configure trusted groups for sudoers"
sudo -H bash -l "$DIR/steps/configure_xsup_aliases.sh"

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC ]]; then
    log_info "Configure hostname aliases for instrument LAN"
    sudo -H bash -l "$DIR/steps/configure_etc_hosts.sh"
    log_info "Configure NFS exports from RTC -> AOC and ICC -> AOC"
    sudo -H bash -l "$DIR/steps/configure_nfs.sh"
fi

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == TIC ]]; then
    log_info "Configure time syncing"
    sudo -H bash -l "$DIR/steps/configure_chrony.sh"
fi

if [[ $MAGAOX_ROLE != ci ]]; then
    log_info "Increase inotify watches (e.g. for VSCode remote users)"
    sudo -H bash -l "$DIR/steps/increase_fs_watcher_limits.sh"
fi

if [[ $MAGAOX_ROLE == AOC ]]; then
    bash -l "$DIR/configure_certificate_renewal.sh"
fi

VENDOR_SOFTWARE_BUNDLE=$DIR/bundle.zip
if [[ ! -e $VENDOR_SOFTWARE_BUNDLE ]]; then
    echo "Couldn't find vendor software bundle at location $VENDOR_SOFTWARE_BUNDLE"
    if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC ]]; then
        log_warn "If this instrument computer will be interfacing with the DMs or framegrabbers, you should Ctrl-C now and get the software bundle."
        read -p "If not, press enter to continue"
    fi
fi



if [[ $MAGAOX_ROLE == AOC ]]; then
    # Configure a tablespace to store postgres data on the /data array
    # and user accounts for the system to use
    bash -l "$DIR/steps/configure_postgresql.sh"
    # Install and enable the service for grafana
    bash -l "$DIR/steps/install_grafana.sh"
fi
# All MagAO-X computers may use the password to connect to the main db
bash -l "$DIR/steps/configure_postgresql_pass.sh"

if [[ $MAGAOX_ROLE == workstation ]]; then
    if [[ $VM_KIND != "wsl" ]]; then
        # Enable forwarding MagAO-X GUIs to the host for VMs
        sudo -H bash -l "$DIR/steps/enable_vm_x11_forwarding.sh"
    fi
    # Install a config in ~/.ssh/config for the vm user
    # to make it easier to make tunnels work
    bash -l "$DIR/steps/configure_ssh_for_workstations.sh" || exit_with_error "Failed to pre-populate SSH config"
fi

## Install proprietary / non-public software
if [[ -e $VENDOR_SOFTWARE_BUNDLE ]]; then
    # Extract bundle
    BUNDLE_TMPDIR=/tmp/vendor_software_bundle_$(date +"%s")
    sudo mkdir -p $BUNDLE_TMPDIR
    sudo unzip -o $VENDOR_SOFTWARE_BUNDLE -d $BUNDLE_TMPDIR
    for vendorname in  alpao andor bmc libhsfw qhyccd teledyne; do
        if [[ ! -d /opt/MagAOX/vendor/$vendorname ]]; then
            sudo cp -R $BUNDLE_TMPDIR/bundle/$vendorname /opt/MagAOX/vendor
        else
            echo "/opt/MagAOX/vendor/$vendorname exists, not overwriting files"
            echo "(but they're in $BUNDLE_TMPDIR/bundle/$vendorname if you want them)"
        fi
    done

    if [[ $MAGAOX_ROLE == RTC ]]; then
        sudo -H bash -l "$DIR/steps/install_alpao.sh"
    fi
    if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == TIC ]]; then
        sudo -H bash -l "$DIR/steps/install_bmc.sh"
    fi
    if [[ $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC ]]; then
        sudo -H bash -l "$DIR/steps/install_libhsfw.sh"
    fi
    if [[ $MAGAOX_ROLE == ICC ]]; then
        sudo -H bash -l "$DIR/steps/install_picam.sh"
        sudo -H bash -l "$DIR/steps/install_kinetix.sh"
    fi
    sudo rm -rf $BUNDLE_TMPDIR
fi

# These steps should work as whatever user is installing, provided
# they are a member of magaox-dev and they have sudo access to install to
# /usr/local. Building as root would leave intermediate build products
# owned by root, which we probably don't want.
cd /opt/MagAOX/source
if [[ $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == TOC ]]; then
    # Initialize the config and calib repos as normal user
    bash -l "$DIR/steps/install_testbed_config.sh"
    bash -l "$DIR/steps/install_testbed_calib.sh"
else
    # Initialize the config and calib repos as normal user
    bash -l "$DIR/steps/install_magao-x_config.sh"
    bash -l "$DIR/steps/install_magao-x_calib.sh"
fi

if [[ $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == AOC ]]; then
    echo "export CGROUPS1_CPUSET_MOUNTPOINT=/opt/MagAOX/cpuset" | sudo tee /etc/profile.d/cgroups1_cpuset_mountpoint.sh
fi

# Create Python env and install Python libs that need special treatment
# Note that subsequent steps will use libs from conda since the base
# env activates by default.
sudo -H bash -l "$DIR/steps/install_python.sh" || exit_with_error "Couldn't install Python"
sudo -H bash -l "$DIR/steps/configure_python.sh" || exit_with_error "Couldn't configure Python environments"
source /opt/conda/bin/activate

# Install first-party deps
bash -l "$DIR/steps/install_milk_and_cacao.sh" || exit_with_error "milk/cacao install failed" # depends on /opt/conda/bin/python existing for plugin build
bash -l "$DIR/steps/install_xrif.sh" || exit_with_error "Failed to build and install xrif"
bash -l "$DIR/steps/install_milkzmq.sh" || exit_with_error "milkzmq install failed"
bash -l "$DIR/steps/install_purepyindi.sh" || exit_with_error "purepyindi install failed"
bash -l "$DIR/steps/install_purepyindi2.sh" || exit_with_error "purepyindi2 install failed"
bash -l "$DIR/steps/install_xconf.sh" || exit_with_error "xconf install failed"
bash -l "$DIR/steps/install_lookyloo.sh" || exit_with_error "lookyloo install failed"
bash -l "$DIR/steps/install_magpyx.sh" || exit_with_error "magpyx install failed"
bash -l "$DIR/steps/install_mxlib.sh" || exit_with_error "Failed to build and install mxlib"
source /etc/profile.d/mxmakefile.sh

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == workstation ]]; then
    # sup web interface
    bash -l "$DIR/steps/install_sup.sh"
fi

## Clone sources to /opt/MagAOX/source/MagAOX
if [[ $MAGAOX_ROLE == ci ]]; then
    ln -sfv ~/project/ /opt/MagAOX/source/MagAOX
else
    log_info "Running as $USER"
    if [[ $DIR != /opt/MagAOX/source/MagAOX/setup ]]; then
        if [[ ! -e /opt/MagAOX/source/MagAOX ]]; then
            echo "Cloning new copy of MagAOX codebase"
            destdir=/opt/MagAOX/source/MagAOX
            git clone $DIR/.. $destdir
            normalize_git_checkout $destdir
            cd $destdir
            # ensure upstream is set somewhere that isn't on the fs to avoid possibly pushing
            # things and not having them go where we expect
            stat /opt/MagAOX/source/MagAOX/.git
            git remote remove origin
            git remote add origin https://github.com/magao-x/MagAOX.git
            git fetch origin
            git branch -u origin/dev dev
            log_success "In the future, you can re-run this script from /opt/MagAOX/source/MagAOX/setup"
            log_info "(In fact, maybe delete $(dirname $DIR)?)"
        else
            cd /opt/MagAOX/source/MagAOX
            git fetch
        fi
    else
        log_info "Running from clone located at $(dirname $DIR), nothing to do for cloning step"
    fi
fi


if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == workstation || $MAGAOX_ROLE == ci ]]; then
    # realtime image viewer
    bash -l "$DIR/steps/install_rtimv.sh" || exit_with_error "Could not install rtimv"
    echo "export RTIMV_CONFIG_PATH=/opt/MagAOX/config" | sudo -H tee /etc/profile.d/rtimv_config_path.sh
fi

# aliases to improve ergonomics of MagAO-X ops
sudo -H bash -l "$DIR/steps/install_aliases.sh"

# CI invokes install_MagAOX.sh as the next step. By separating the
# real build into another step, we can cache the slow provisioning
# steps and reuse them on subsequent runs.
if [[ -z $CI ]]; then
    cd /opt/MagAOX/source/MagAOX
    bash -l "$DIR/steps/install_MagAOX.sh" || exit 1
fi

if [[ $MAGAOX_ROLE != ci && $MAGAOX_ROLE != container && $MAGAOX_ROLE != vm ]]; then
    sudo -H bash -l "$DIR/steps/configure_startup_services.sh"

    log_info "Generating subuid and subgid files, may need to run podman system migrate"
    sudo -H python "$DIR/generate_subuid_subgid.py" || exit_with_error "Generating subuid/subgid files for podman failed"
    sudo -H podman system migrate || exit_with_error "Could not run podman system migrate"
fi

log_success "Provisioning complete"

if [[ $MAGAOX_ROLE == ci || $MAGAOX_ROLE == container ]]; then
    exit 0
elif [[ -z "$(groups | grep magaox)" ]]; then
    log_info "You now need to log out and back in for group changes to take effect"
else
    log_info "You'll probably want to run"
    log_info "    source /etc/profile.d/*.sh"
    log_info "to get all the new environment variables set."
fi
