#!/bin/bash
set -o pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
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
# Get just the XX beginning of a XX.YY version string
MAJOR_VERSION=${VERSION_ID%.*}

roleScript=/etc/profile.d/magaox_role.sh
VM_KIND=$(systemd-detect-virt)
if [[ ! $VM_KIND == "none" ]]; then
    echo "Detected virtualization: $VM_KIND"
    if [[ ! -z $CI ]]; then
        echo "export MAGAOX_ROLE=ci" | $_REAL_SUDO tee $roleScript
    elif [[ ! -e $roleScript ]]; then
        echo "export MAGAOX_ROLE=vm" | $_REAL_SUDO tee $roleScript
    fi
fi
if [[ ! -e $roleScript ]]; then
    echo "Export \$MAGAOX_ROLE in $roleScript first"
    exit 1
fi
source $roleScript

if [[ $MAGAOX_ROLE == ci ]]; then
    export NEEDRESTART_SUSPEND=yes
    export DEBIAN_FRONTEND=noninteractive
    cat <<'HERE' | sudo tee /etc/profile.d/ci.sh || exit 1
export NEEDRESTART_SUSPEND=yes
export DEBIAN_FRONTEND=noninteractive
HERE
fi

# Get logging functions
source $DIR/_common.sh

# Install OS packages first
osPackagesScript="$DIR/steps/install_${ID}_${MAJOR_VERSION}_packages.sh"
$_REAL_SUDO -H bash -l $osPackagesScript || exit_with_error "Failed to install packages from $osPackagesScript"

if [[ $ID == centos ]]; then
    $_REAL_SUDO -H bash -l "$DIR/steps/install_cmake.sh" || exit 1
fi

distroSpecificScript="$DIR/steps/configure_${ID}_${MAJOR_VERSION}.sh"
$_REAL_SUDO -H bash -l $distroSpecificScript || exit_with_error "Failed to configure ${ID} from $distroSpecificScript"

if [[ $VM_KIND != "none" ]]; then
    git config --global --replace-all safe.directory '*'
    sudo -H git config --global --replace-all safe.directory '*'
fi

bash -l "$DIR/steps/configure_trusted_sudoers.sh" || exit_with_error "Could not configure trusted groups for sudoers"
sudo -H bash -l "$DIR/steps/configure_xsup_aliases.sh"

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC ]]; then
    # Configure hostname aliases for instrument LAN
    sudo -H bash -l "$DIR/steps/configure_etc_hosts.sh"
    # Configure NFS exports from RTC -> AOC and ICC -> AOC
    sudo -H bash -l "$DIR/steps/configure_nfs.sh"
fi

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == TIC ]]; then
    # Configure time syncing
    sudo -H bash -l "$DIR/steps/configure_chrony.sh"
fi

if [[ $MAGAOX_ROLE != ci ]]; then
    # Increase inotify watches
    sudo -H bash -l "$DIR/steps/increase_fs_watcher_limits.sh"
fi

# The VM and CI provisioning doesn't run setup_users_and_groups.sh
# separately as in the instrument instructions; we have to run it
if [[ $MAGAOX_ROLE == vm || $MAGAOX_ROLE == ci ]]; then
    bash -l "$DIR/setup_users_and_groups.sh"
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

## Set up file structure and permissions
sudo -H bash -l "$DIR/steps/ensure_dirs_and_perms.sh" $MAGAOX_ROLE

if [[ $MAGAOX_ROLE == AOC ]]; then
    # Configure a tablespace to store postgres data on the /data array
    # and user accounts for the system to use
    bash -l "$DIR/steps/configure_postgresql.sh"
    # Install and enable the service for grafana
    bash -l "$DIR/steps/install_grafana.sh"
fi
# All MagAO-X computers may use the password to connect to the main db
bash -l "$DIR/steps/configure_postgresql_pass.sh"

if [[ $MAGAOX_ROLE == vm ]]; then
    if [[ $VM_KIND != "wsl" ]]; then
        # Enable forwarding MagAO-X GUIs to the host for VMs
        sudo -H bash -l "$DIR/steps/enable_vm_x11_forwarding.sh"
    fi
    # Install a config in ~/.ssh/config for the vm user
    # to make it easier to make tunnels work
    bash -l "$DIR/steps/configure_vm_ssh.sh" || exit_with_error "Failed to set up VM SSH"
fi

# Install dependencies for the GUIs
if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == ci || $MAGAOX_ROLE == vm || $MAGAOX_ROLE == workstation ]]; then
    sudo -H bash -l "$DIR/steps/install_gui_dependencies.sh"
fi

# Install Linux headers (instrument computers use the RT kernel / headers)
if [[ $MAGAOX_ROLE == ci || $MAGAOX_ROLE == vm || $MAGAOX_ROLE == workstation || $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC ]]; then
    if [[ $ID == ubuntu ]]; then
        sudo -i apt install -y linux-headers-generic
    elif [[ $ID == centos ]]; then
        sudo yum install -y kernel-devel-$(uname -r) || sudo yum install -y kernel-devel
    fi
fi
## Build third-party dependencies under /opt/MagAOX/vendor
cd /opt/MagAOX/vendor
sudo -H bash -l "$DIR/steps/install_rclone.sh" || exit 1
bash -l "$DIR/steps/install_openblas.sh" || exit 1
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TIC ]]; then
    bash -l "$DIR/steps/install_cuda_rocky_9.sh" || exit_with_error "CUDA install failed"
fi
sudo -H bash -l "$DIR/steps/install_fftw.sh" || exit 1
sudo -H bash -l "$DIR/steps/install_cfitsio.sh" || exit 1
sudo -H bash -l "$DIR/steps/install_eigen.sh" || exit 1
sudo -H bash -l "$DIR/steps/install_zeromq.sh" || exit 1
sudo -H bash -l "$DIR/steps/install_cppzmq.sh" || exit 1
sudo -H bash -l "$DIR/steps/install_flatbuffers.sh" || exit 1
if [[ $MAGAOX_ROLE == AOC ]]; then
    sudo -H bash -l "$DIR/steps/install_lego.sh"
fi
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == ci || ( $MAGAOX_ROLE == vm && $VM_KIND == vagrant ) ]]; then
    sudo -H bash -l "$DIR/steps/install_basler_pylon.sh"
fi
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == ci || ( $MAGAOX_ROLE == vm && $VM_KIND == vagrant ) ]]; then
    sudo -H bash -l "$DIR/steps/install_edt.sh"
fi

# SuSE packages need either Python 3.6 or 3.10, but Rocky 9.2 has Python 3.9 as /bin/python, so we build our own RPM:
if [[ $ID == rocky && $MAGAOX_ROLE != container ]]; then
  sudo -H bash -l "$DIR/steps/install_cpuset.sh" || exit_with_error "Couldn't install cpuset from source"
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
    # Note that 'vm' is in the list for ease of testing the install_* scripts
    if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == vm ]]; then
        if [[ $ID == centos ]]; then
            sudo -H bash -l "$DIR/steps/install_alpao.sh"
        fi
        sudo -H bash -l "$DIR/steps/install_andor.sh"
    fi
    if [[ $ID == centos && ( $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == vm ) ]]; then
        sudo -H bash -l "$DIR/steps/install_bmc.sh"
    fi
    if [[ $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == vm ]]; then
        sudo -H bash -l "$DIR/steps/install_libhsfw.sh"
    fi
    if [[ $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == vm ]]; then
        sudo -H bash -l "$DIR/steps/install_picam.sh"
        sudo -H bash -l "$DIR/steps/install_kinetix.sh"
    fi
    sudo rm -rf $BUNDLE_TMPDIR
fi

# These steps should work as whatever user is installing, provided
# they are a member of magaox-dev and they have sudo access to install to
# /usr/local. Building as root would leave intermediate build products
# owned by root, which we probably don't want.
#
# On a Vagrant VM, we need to "sudo" to become vagrant since the provisioning
# runs as root.
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
sudo -H bash -l "$DIR/steps/install_python.sh"
sudo -H bash -l "$DIR/steps/configure_python.sh"
source /opt/conda/bin/activate

if [[ $ID == centos && ( $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == vm ) ]]; then
    sudo -H mamba install -y qwt qt=5.9.7 || exit 1
    log_info "Installed qwt from conda for widgeting purposes on old CentOS"
fi

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

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == vm ||  $MAGAOX_ROLE == workstation ]]; then
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


if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == vm || $MAGAOX_ROLE == workstation || $MAGAOX_ROLE == ci ]]; then
    # realtime image viewer
    bash -l "$DIR/steps/install_rtimv.sh" || exit_with_error "Could not install rtimv"
    echo "export RTIMV_CONFIG_PATH=/opt/MagAOX/config" | sudo -H tee /etc/profile.d/rtimv_config_path.sh
fi

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == vm ||  $MAGAOX_ROLE == workstation ]]; then
    # regular old ds9 image viewer
    sudo -H bash -l "$DIR/steps/install_ds9.sh"
fi

# aliases to improve ergonomics of MagAO-X ops
sudo -H bash -l "$DIR/steps/install_aliases.sh"

# CI invokes install_MagAOX.sh as the next step (see .circleci/config.yml)
# By separating the real build into another step, we can cache the slow provisioning steps
# and reuse them on subsequent runs.
if [[ $MAGAOX_ROLE != ci ]]; then
    cd /opt/MagAOX/source/MagAOX
    bash -l "$DIR/steps/install_MagAOX.sh" || exit 1
fi

if [[ $MAGAOX_ROLE != ci && $MAGAOX_ROLE != container && $MAGAOX_ROLE != vm ]]; then
    sudo -H bash -l "$DIR/steps/configure_startup_services.sh"

    log_info "Generating subuid and subgid files, may need to run podman system migrate"
    sudo -H python "$DIR/generate_subuid_subgid.py" || exit_with_error "Generating subuid/subgid files for podman failed"
    sudo -H podman system migrate || exit_with_error "Could not run podman system migrate"

    # To try and debug hardware issues, ICC and RTC replicate their
    # kernel console log over UDP to AOC over the instrument LAN.
    # The script that collects these messages is in ../scripts/netconsole_logger
    # so we have to install its service unit after 'make scripts_install'
    # runs.
    sudo -H bash -l "$DIR/steps/configure_kernel_netconsole.sh"
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
