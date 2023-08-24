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

# Get logging functions
# Also defines $ID and $VERSION_ID, from /etc/os-release, so we can detect which distribution we're on
source $DIR/_common.sh

# Install OS packages first
osPackagesScript="$DIR/steps/install_${ID}_${VERSION_ID}_packages.sh"
$_REAL_SUDO bash -l $osPackagesScript || exit_error "Failed to install packages from $osPackagesScript"

if [[ $ID == centos ]]; then
    $_REAL_SUDO bash -l "$DIR/steps/install_cmake.sh" || exit 1
fi

distroSpecificScript="$DIR/steps/configure_${ID}_${VERSION_ID}.sh"
$_REAL_SUDO bash -l $distroSpecificScript || exit_error "Failed to configure ${ID} from $distroSpecificScript"

if [[ $VM_KIND != "none" ]]; then
    git config --global --replace-all safe.directory '*'
    sudo git config --global --replace-all safe.directory '*'
fi

sudo bash -l "$DIR/steps/configure_xsup_aliases.sh"

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC ]]; then
    # Configure hostname aliases for instrument LAN
    sudo bash -l "$DIR/steps/configure_etc_hosts.sh"
    # Configure NFS exports from RTC -> AOC and ICC -> AOC
    sudo bash -l "$DIR/steps/configure_nfs.sh"
fi

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == TIC ]]; then
    # Configure time syncing
    sudo bash -l "$DIR/steps/configure_chrony.sh"
fi

if [[ $MAGAOX_ROLE != ci ]]; then
    # Increase inotify watches
    sudo bash -l "$DIR/steps/increase_fs_watcher_limits.sh"
fi

# The VM and CI provisioning doesn't run setup_users_and_groups.sh
# separately as in the instrument instructions; we have to run it
if [[ $MAGAOX_ROLE == vm || $MAGAOX_ROLE == ci ]]; then
    bash -l "$DIR/setup_users_and_groups.sh"
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
sudo bash -l "$DIR/steps/ensure_dirs_and_perms.sh" $MAGAOX_ROLE

if [[ $MAGAOX_ROLE == vm ]]; then
    if [[ $VM_KIND != "wsl" ]]; then
        # Enable forwarding MagAO-X GUIs to the host for VMs
        sudo bash -l "$DIR/steps/enable_vm_x11_forwarding.sh"
    fi
    # Install a config in ~/.ssh/config for the vm user
    # to make it easier to make tunnels work
    bash -l "$DIR/steps/configure_vm_ssh.sh" || exit_error "Failed to set up VM SSH"
fi

# Install dependencies for the GUIs
if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == ci || $MAGAOX_ROLE == vm || $MAGAOX_ROLE == workstation ]]; then
    sudo bash -l "$DIR/steps/install_gui_dependencies.sh"
fi

# Install Linux headers (instrument computers use the RT kernel / headers)
if [[ $MAGAOX_ROLE == ci || $MAGAOX_ROLE == vm || $MAGAOX_ROLE == workstation || $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC ]]; then
    if [[ $ID == ubuntu ]]; then
        sudo NEEDRESTART_SUSPEND=yes apt install -y linux-headers-generic
    elif [[ $ID == centos ]]; then
        sudo yum install -y kernel-devel-$(uname -r) || sudo yum install -y kernel-devel
    fi
fi
## Build third-party dependencies under /opt/MagAOX/vendor
cd /opt/MagAOX/vendor
sudo bash -l "$DIR/steps/install_rclone.sh" || exit 1
if grep -q "GenuineIntel" /proc/cpuinfo; then
    if [[ $ID == "ubuntu" ]]; then
        sudo bash -l "$DIR/steps/install_mkl_package.sh" || exit 1
    else
        ###sudo bash -l "$DIR/steps/install_mkl_tarball.sh" || exit 1
        sudo bash -l "$DIR/steps/install_mkl_tarball.sh" || ( echo "Failed MKL install; continuing" 1>&2 || exit 0 )
    fi
    export BLAS_VENDOR=intel
    source /etc/profile.d/mklvars.sh
else
    export BLAS_VENDOR=openblas
fi
if grep -q "GenuineIntel" /proc/cpuinfo; then
    if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == ci ]]; then
        bash -l "$DIR/steps/install_cuda.sh" || exit_error "CUDA install failed"
        sudo bash -l "$DIR/steps/install_magma.sh" || exit_error "MAGMA install failed"
    fi
fi
sudo bash -l "$DIR/steps/install_fftw.sh" || exit 1
sudo bash -l "$DIR/steps/install_cfitsio.sh" || exit 1
sudo bash -l "$DIR/steps/install_eigen.sh" || exit 1
sudo bash -l "$DIR/steps/install_zeromq.sh" || exit 1
sudo bash -l "$DIR/steps/install_cppzmq.sh" || exit 1
sudo bash -l "$DIR/steps/install_flatbuffers.sh" || exit 1
if [[ $MAGAOX_ROLE == AOC ]]; then
    sudo bash -l "$DIR/steps/install_lego.sh"
fi
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == ci || ( $MAGAOX_ROLE == vm && $VM_KIND == vagrant ) ]]; then
    sudo bash -l "$DIR/steps/install_basler_pylon.sh"
fi
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == ci || ( $MAGAOX_ROLE == vm && $VM_KIND == vagrant ) ]]; then
    sudo bash -l "$DIR/steps/install_edt.sh"
fi

# SuSE packages need either Python 3.6 or 3.10, but Rocky 9.2 has Python 3.9 as /bin/python, so we build our own RPM:
if [[ $ID == rocky ]]; then
  sudo bash -l "$DIR/steps/install_cpuset.sh" || exit_error "Couldn't install cpuset from source"
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
            sudo bash -l "$DIR/steps/install_alpao.sh"
        fi
        sudo bash -l "$DIR/steps/install_andor.sh"
    fi
    if [[ $ID == centos && ( $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == vm ) ]]; then
        sudo bash -l "$DIR/steps/install_bmc.sh"
    fi
    if [[ $MAGAOX_ROLE == ICC ]]; then
        sudo bash -l "$DIR/steps/install_libhsfw.sh"
        sudo bash -l "$DIR/steps/install_picam.sh"
        sudo bash -l "$DIR/steps/install_kinetix.sh"
    fi
    sudo rm -rf $BUNDLE_TMPDIR
fi

## Build first-party dependencies
cd /opt/MagAOX/source
bash -l "$DIR/steps/install_xrif.sh" || exit_error "Failed to build and install xrif"
bash -l "$DIR/steps/install_mxlib.sh" || exit_error "Failed to build and install mxlib"
source /etc/profile.d/mxmakefile.sh

## Build MagAO-X and install sources to /opt/MagAOX/source/MagAOX
if [[ $MAGAOX_ROLE == ci ]]; then
    ln -sfv ~/project/ /opt/MagAOX/source/MagAOX
else
    log_info "Running as $USER"
    if [[ $DIR != /opt/MagAOX/source/MagAOX/setup ]]; then
        if [[ ! -e /opt/MagAOX/source/MagAOX ]]; then
            echo "Cloning new copy of MagAOX codebase"
            orgname=magao-x
            reponame=MagAOX
            parentdir=/opt/MagAOX/source
            clone_or_update_and_cd $orgname $reponame $parentdir
            if git --git-dir="$parentdir/$reponame/.git" remote -v 2>/dev/null | grep -q "^origin  *https://.*/$orgname/$reponame" ; then
                # ensure upstream is set somewhere that isn't on the fs to avoid possibly pushing
                # things and not having them go where we expect
                stat /opt/MagAOX/source/MagAOX/.git
                git remote remove origin
                git remote add origin https://github.com/magao-x/MagAOX.git
                git fetch origin
                git branch -u origin/dev dev
            fi
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
# These steps should work as whatever user is installing, provided
# they are a member of $instrument_dev_group and they have sudo access to install to
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

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == vm ]]; then
    echo "export RTIMV_CONFIG_PATH=/opt/MagAOX/config" | sudo tee /etc/profile.d/rtimv_config_path.sh
fi

if [[ $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == AOC ]]; then
    echo "export CGROUPS1_CPUSET_MOUNTPOINT=/opt/MagAOX/cpuset" | sudo tee /etc/profile.d/cgroups1_cpuset_mountpoint.sh
fi

# Create Python env and install Python libs that need special treatment
# Note that subsequent steps will use libs from conda since the base
# env activates by default.
sudo bash -l "$DIR/steps/install_python.sh"
sudo bash -l "$DIR/steps/configure_python.sh"
source /opt/conda/bin/activate

if [[ $ID == centos && ( $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == vm ) ]]; then
    sudo mamba install -y qwt qt=5.9.7 || exit 1
    log_info "Installed qwt from conda for widgeting purposes on old CentOS"
fi

# Install first-party deps
bash -l "$DIR/steps/install_milk_and_cacao.sh" || exit_error "milk/cacao install failed" # depends on /opt/conda/bin/python existing for plugin build
bash -l "$DIR/steps/install_milkzmq.sh" || exit_error "milkzmq install failed"
bash -l "$DIR/steps/install_purepyindi.sh" || exit_error "purepyindi install failed"
bash -l "$DIR/steps/install_magpyx.sh" || exit_error "magpyx install failed"


# TODO:jlong: uncomment when it's back in working order
# if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == vm ||  $MAGAOX_ROLE == workstation ]]; then
#     # sup web interface
#     bash -l "$DIR/steps/install_sup.sh"
# fi

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == vm || $MAGAOX_ROLE == workstation || $MAGAOX_ROLE == ci ]]; then
    # realtime image viewer
    bash -l "$DIR/steps/install_rtimv.sh" || exit 1
fi

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == vm ||  $MAGAOX_ROLE == workstation ]]; then
    # regular old ds9 image viewer
    sudo bash -l "$DIR/steps/install_ds9.sh"
fi

# aliases to improve ergonomics of MagAO-X ops
sudo bash -l "$DIR/steps/install_aliases.sh"

# CircleCI invokes install_MagAOX.sh as the next step (see .circleci/config.yml)
# By separating the real build into another step, we can cache the slow provisioning steps
# and reuse them on subsequent runs.
if [[ $MAGAOX_ROLE != ci ]]; then
    bash -l "$DIR/steps/install_MagAOX.sh" || exit 1
fi

sudo bash -l "$DIR/steps/configure_startup_services.sh"

if which podman ; then
  log_info "Generating subuid and subgid files, may need to run podman system migrate"
  sudo python "$DIR/generate_subuid_subgid.py" || exit_error "Generating subuid/subgid files for podman failed"
  sudo podman system migrate || exit_error "Could not run podman system migrate"
fi

# To try and debug hardware issues, ICC and RTC replicate their
# kernel console log over UDP to AOC over the instrument LAN.
# The script that collects these messages is in ../scripts/netconsole_logger
# so we have to install its service unit after 'make scripts_install'
# runs.
sudo bash -l "$DIR/steps/configure_kernel_netconsole.sh"

log_success "Provisioning complete"
if [[ $MAGAOX_ROLE != vm && $MAGAOX_ROLE != ci && $MAGAOX_ROLE != container ]]; then
    log_info "You'll probably want to run"
    log_info "    source /etc/profile.d/*.sh"
    log_info "to get all the new environment variables set."
fi
