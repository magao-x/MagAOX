#!/bin/bash
set -eo pipefail
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
# Defines $ID and $VERSION_ID so we can detect which distribution we're on
source /etc/os-release
if [[ $ID == ubuntu ]]; then
    # Stop annoying messages about messages
    # https://superuser.com/questions/1160025/how-to-solve-ttyname-failed-inappropriate-ioctl-for-device-in-vagrant
    sudo sed -i -e 's/mesg n .*true/tty -s \&\& mesg n/g' ~/.profile
fi
# Detect whether we're running in some kind of VM or container
if [[ -d /vagrant || $CI == true ]]; then
    if [[ -d /vagrant ]]; then
        DIR="/vagrant/setup"
        # If set already (i.e. not the first time we ran this),
        # don't override what's in the environment
        if [[ -z $MAGAOX_ROLE ]]; then
            MAGAOX_ROLE=vm
        else
            echo "Would have set MAGAOX_ROLE=vm, but already have MAGAOX_ROLE=$MAGAOX_ROLE"
        fi
        CI=false
    else
        if [[ -z $MAGAOX_ROLE ]]; then
            MAGAOX_ROLE=ci
        else
            echo "Would have set MAGAOX_ROLE=ci, but already have MAGAOX_ROLE=$MAGAOX_ROLE"
        fi
    fi
else
    # Only bother prompting if no role was specified as the command line arg to provision.sh
    if [[ ! -z "$1" ]]; then
        MAGAOX_ROLE="$1"
    fi
    if [[ -z $MAGAOX_ROLE ]]; then
        MAGAOX_ROLE=""
        echo "Choose the role for this machine"
        echo "    AOC - Adaptive optics Operator Computer"
        echo "    RTC - Real Time control Computer"
        echo "    ICC - Instrument Control Computer"
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
                workstation)
                    MAGAOX_ROLE=workstation
                    ;;
                *)
                    echo "Must be one of AOC, RTC, ICC, or workstation."
                    continue
            esac
        done
    else
        echo "Already have MAGAOX_ROLE=$MAGAOX_ROLE, not prompting for it. (Edit /etc/profile.d/magaox_role.sh if it's wrong)"
    fi
    VAGRANT=false
    CI=false
    set +e; groups | grep magaox-dev; set -e
    not_in_group=$?
    if [[ "$EUID" == 0 || $not_in_group != 0 ]]; then
        echo "This script should be run as a normal user"
        echo "in the magaox-dev group with sudo access, not root."
        echo "Run $DIR/setup_users_and_groups.sh first."
        exit 1
    fi
    # Prompt for sudo authentication
    $_REAL_SUDO -v
    # Keep the sudo timestamp updated until this script exits
    while true; do $_REAL_SUDO -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
fi

if [[ ! -z "$1" ]]; then
    case $1 in
        AOC)
            MAGAOX_ROLE=AOC
            ;;
        RTC)
            MAGAOX_ROLE=RTC
            ;;
        ICC)
            MAGAOX_ROLE=ICC
            ;;
        workstation)
            MAGAOX_ROLE=workstation
            ;;
        *)
            echo "Argument must be one of AOC, RTC, ICC, or workstation."
            exit 1
    esac
fi
echo "Starting '$MAGAOX_ROLE' provisioning"
echo "export MAGAOX_ROLE=$MAGAOX_ROLE" | sudo tee /etc/profile.d/magaox_role.sh
export MAGAOX_ROLE

# Shouldn't be any more undefined variables after (maybe) $1,
# so tell bash to die if it encounters any
set -u

# Get logging functions
source $DIR/_common.sh

# Install OS packages first
if [[ $ID == ubuntu ]]; then
    sudo bash -l "$DIR/steps/install_ubuntu_bionic_packages.sh"
elif [[ $ID == centos && $VERSION_ID == 7 ]]; then
    sudo bash -l "$DIR/steps/install_centos7_packages.sh"
    sudo bash -l "$DIR/steps/install_devtoolset-7.sh"
else
    log_error "No special casing for $ID $VERSION_ID yet, abort"
    exit 1
fi

# Configure hostname aliases and time synchronization
if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC ]]; then
    sudo bash -l "$DIR/steps/configure_etc_hosts.sh"
    sudo bash -l "$DIR/steps/configure_chrony.sh"
fi

# Configure executable search path
sudo bash -l "$DIR/steps/put_usr_local_bin_on_path.sh"

# The VM and CI provisioning doesn't run setup_users_and_groups.sh
# separately as in the instrument instructions; we have to run it
if [[ $MAGAOX_ROLE == vm || $MAGAOX_ROLE == ci ]]; then
    sudo bash -l "$DIR/setup_users_and_groups.sh"
fi

VENDOR_SOFTWARE_BUNDLE=$DIR/bundle.zip
if [[ ! -e $VENDOR_SOFTWARE_BUNDLE ]]; then
    echo "Couldn't find vendor software bundle at location $VENDOR_SOFTWARE_BUNDLE"
    echo "(Generate with ~/Box/MagAO-X/Vendor\ Software/generate_bundle.sh)"
    if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC ]]; then
        log_warn "If this instrument computer will be interfacing with the DMs or framegrabbers, you should Ctrl-C now and get the software bundle."
        read -p "If not, press enter to continue"
    fi
fi

## Set up file structure and permissions
sudo bash -l "$DIR/steps/ensure_dirs_and_perms.sh" $MAGAOX_ROLE

# Enable forwarding MagAO-X GUIs to the host for VMs
if [[ $MAGAOX_ROLE == vm ]]; then
    sudo bash -l "$DIR/steps/enable_vm_x11_forwarding.sh"
fi

# Install dependencies for the GUIs
if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ci || $MAGAOX_ROLE == vm ||  $MAGAOX_ROLE == workstation ]]; then
    sudo bash -l "$DIR/steps/install_gui_dependencies.sh"
fi

# Install Linux headers (instrument computers use the RT kernel / headers)
if [[ $MAGAOX_ROLE == ci || $MAGAOX_ROLE == vm || $MAGAOX_ROLE == workstation || $MAGAOX_ROLE == AOC ]]; then
    if [[ $ID == ubuntu ]]; then
        sudo apt install -y linux-headers-generic
    elif [[ $ID == centos ]]; then
        sudo yum install -y kernel-devel-$(uname -r) || yum install -y kernel-devel
    fi
fi
## Build third-party dependencies under /opt/MagAOX/vendor
cd /opt/MagAOX/vendor
sudo bash -l "$DIR/steps/install_mkl_tarball.sh"
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ci ]]; then
    sudo bash -l "$DIR/steps/install_cuda.sh"
    sudo bash -l "$DIR/steps/install_magma.sh"
fi
sudo bash -l "$DIR/steps/install_fftw.sh"
sudo bash -l "$DIR/steps/install_cfitsio.sh"
sudo bash -l "$DIR/steps/install_sofa.sh"
sudo bash -l "$DIR/steps/install_xpa.sh"
sudo bash -l "$DIR/steps/install_eigen.sh"
sudo bash -l "$DIR/steps/install_cppzmq.sh"
sudo bash -l "$DIR/steps/install_levmar.sh"
sudo bash -l "$DIR/steps/install_flatbuffers.sh"
sudo bash -l "$DIR/steps/install_xrif.sh"
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC || "$MAGAOX_ROLE" == "ci" ]]; then
    sudo bash -l "$DIR/steps/install_basler_pylon.sh"
    sudo bash -l "$DIR/steps/install_edt.sh"
    sudo bash -l "$DIR/steps/install_picam.sh"
fi

## Install proprietary / non-public software
if [[ -e $VENDOR_SOFTWARE_BUNDLE ]]; then
    # Extract bundle
    BUNDLE_TMPDIR=/tmp/vendor_software_bundle_$(date +"%s")
    sudo mkdir -p $BUNDLE_TMPDIR
    sudo unzip -o $VENDOR_SOFTWARE_BUNDLE -d $BUNDLE_TMPDIR
    for vendorname in alpao bmc andor; do
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
    if [[ $ID == centos && ( $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == vm ) ]]; then
        sudo bash -l "$DIR/steps/install_bmc.sh"
    fi
    sudo rm -rf $BUNDLE_TMPDIR
fi

## Build first-party dependencies
cd /opt/MagAOX/source
sudo bash -l "$DIR/steps/install_mxlib.sh"
source /etc/profile.d/mxmakefile.sh

## Build MagAO-X and install sources to /opt/MagAOX/source/MagAOX
MAYBE_SUDO=
if [[ $MAGAOX_ROLE == vm ]]; then
    MAYBE_SUDO="$_REAL_SUDO -u vagrant"
    # Create or replace symlink to sources so we develop on the host machine's copy
    # (unlike prod, where we install a new clone of the repo to this location)
    sudo ln -nfs /vagrant /opt/MagAOX/source/MagAOX
    cd /opt/MagAOX/source/MagAOX
    log_success "Symlinked /opt/MagAOX/source/MagAOX to /vagrant (host folder)"
    sudo usermod -G magaox,magaox-dev vagrant
    log_success "Added vagrant user to magaox,magaox-dev"
elif [[ $MAGAOX_ROLE == ci ]]; then
    ln -sfv ~/project/ /opt/MagAOX/source/MagAOX
else
    if [[ $DIR != /opt/MagAOX/source/MagAOX/setup ]]; then
        if [[ ! -e /opt/MagAOX/source/MagAOX ]]; then
            echo "Cloning new copy of MagAOX codebase"
            orgname=magao-x
            reponame=MagAOX
            parentdir=/opt/MagAOX/source/
            clone_or_update_and_cd $orgname $reponame $parentdir
            # ensure upstream is set somewhere that isn't on the fs to avoid possibly pushing
            # things and not having them go where we expect
            git remote remove origin
            git remote add origin https://github.com/magao-x/MagAOX.git
            git fetch origin
            git branch -u origin/master master
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
# they are a member of magaox-dev and they have sudo access to install to
# /usr/local. Building as root would leave intermediate build products
# owned by root, which we probably don't want.
#
# On a Vagrant VM, we need to "sudo" to become vagrant since the provisioning
# runs as root.
cd /opt/MagAOX/source
# Initialize the config and calib repos as normal user
$MAYBE_SUDO bash -l "$DIR/steps/install_magao-x_config.sh"
$MAYBE_SUDO bash -l "$DIR/steps/install_magao-x_calib.sh"
# Install first-party deps
$MAYBE_SUDO bash -l "$DIR/steps/install_cacao.sh"
$MAYBE_SUDO bash -l "$DIR/steps/install_milkzmq.sh"
# Create Python env and install Python libs that need special treatment
# (Depends on presence of /opt/MagAOX/config/conda_env_py37.yml)
# Note that subsequent steps will use libs from conda since the base
# env activates by default.
sudo bash -l "$DIR/steps/install_python.sh"
sudo bash -l "$DIR/steps/create_conda_envs.sh"
sudo bash -l "$DIR/steps/install_purepyindi.sh"
sudo bash -l "$DIR/steps/install_sup.sh"
sudo bash -l "$DIR/steps/install_magpyx.sh"
sudo bash -l "$DIR/steps/install_imagestreamio_python.sh"

# AOC, vm, and workstation should all install rtimv
if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == vm ||  $MAGAOX_ROLE == workstation ]]; then
    sudo bash -l "$DIR/steps/install_rtimv.sh"
fi

# CircleCI invokes install_MagAOX.sh as the next step (see .circleci/config.yml)
# By separating the real build into another step, we can cache the slow provisioning steps
# and reuse them on subsequent runs.
if [[ $MAGAOX_ROLE != ci ]]; then
    $MAYBE_SUDO bash -l "$DIR/steps/install_MagAOX.sh"
fi

log_success "Provisioning complete"
if [[ $MAGAOX_ROLE != vm ]]; then
    log_info "You'll probably want to run"
    log_info "    source /etc/profile.d/*.sh"
    log_info "to get all the new environment variables set."
fi
