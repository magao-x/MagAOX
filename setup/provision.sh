#!/bin/bash
set -eo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ -e /usr/bin/sudo ]]; then
  _REAL_SUDO=/usr/bin/sudo
elif [[ -e /bin/sudo ]]; then
  _REAL_SUDO=/bin/sudo
else
  _REAL_SUDO=$(which sudo)
fi
if [[ -d /vagrant || $CI == true ]]; then
    TARGET_ENV=vm
    if [[ -d /vagrant ]]; then
        DIR="/vagrant/setup"
        VAGRANT=true
        CI=false
    else
        VAGRANT=false
    fi
    echo "Setting up for VM use"
    sudo bash -l "$DIR/setup_users_and_groups.sh"
    sudo yum install -y kernel-devel-$(uname -r) || sudo yum install -y kernel-devel
else
    TARGET_ENV=instrument
    VAGRANT=false
    CI=false
    echo "Setting up for instrument use"
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
source $DIR/_common.sh

VENDOR_SOFTWARE_BUNDLE=$DIR/vendor_software_bundle.tar.gz
if [[ ! -e $VENDOR_SOFTWARE_BUNDLE ]]; then
    echo "Couldn't find vendor software bundle at location $VENDOR_SOFTWARE_BUNDLE"
    echo "(Generate with ~/Box/MagAO-X/Vendor\ Software/generate_bundle.sh)"
    if [[ $TARGET_ENV == "instrument" ]]; then
        exit 1
    fi
fi
# die on uninitialized variables (typo guard)
set -u

## Set up file structure and permissions
sudo bash -l "$DIR/steps/ensure_dirs_and_perms.sh" $TARGET_ENV

## Build third-party dependencies under /opt/MagAOX/vendor
cd /opt/MagAOX/vendor
sudo bash -l "$DIR/steps/install_os_packages.sh"
sudo bash -l "$DIR/steps/install_devtoolset-7.sh"
sudo bash -l "$DIR/steps/install_mkl_tarball.sh"
sudo bash -l "$DIR/steps/install_cuda.sh"
sudo bash -l "$DIR/steps/install_fftw.sh"
sudo bash -l "$DIR/steps/install_cfitsio.sh"
sudo bash -l "$DIR/steps/install_sofa.sh"
sudo bash -l "$DIR/steps/install_xpa.sh"
sudo bash -l "$DIR/steps/install_eigen.sh"
sudo bash -l "$DIR/steps/install_cppzmq.sh"
sudo bash -l "$DIR/steps/install_levmar.sh"
sudo bash -l "$DIR/steps/install_flatbuffers.sh"
sudo bash -l "$DIR/steps/install_xrif.sh"
if ! $VAGRANT; then
    sudo bash -l "$DIR/steps/install_magma.sh"
fi
sudo bash -l "$DIR/steps/install_basler_pylon.sh"
sudo bash -l "$DIR/steps/install_edt.sh"
sudo bash -l "$DIR/steps/install_picam.sh"

## Install proprietary / non-public software
if [[ -e $VENDOR_SOFTWARE_BUNDLE ]]; then
    # Extract bundle
    BUNDLE_TMPDIR=/tmp/vendor_software_bundle
    mkdir -p /tmp/vendor_software_bundle
    tar xzf $VENDOR_SOFTWARE_BUNDLE -C $BUNDLE_TMPDIR
    for vendorname in alpao bmc; do
        if [[ ! -d /opt/MagAOX/vendor/$vendorname ]]; then
            sudo cp -R $BUNDLE_TMPDIR/$vendorname /opt/MagAOX/vendor
        else
            echo "/opt/MagAOX/vendor/$vendorname exists, not overwriting files"
            echo "(but they're in $BUNDLE_TMPDIR/$vendorname if you want them)"
        fi
    done
    sudo bash -l "$DIR/steps/install_alpao.sh"
    sudo bash -l "$DIR/steps/install_bmc.sh"
fi

## Build first-party dependencies
cd /opt/MagAOX/source
sudo bash -l "$DIR/steps/install_mxlib.sh"
source /etc/profile.d/mxmakefile.sh

## Build MagAO-X and install sources to /opt/MagAOX/source/MagAOX
MAYBE_SUDO=
if $VAGRANT; then
    MAYBE_SUDO="$_REAL_SUDO -u vagrant"
    # Create or replace symlink to sources so we develop on the host machine's copy
    # (unlike prod, where we install a new clone of the repo to this location)
    ln -nfs /vagrant /opt/MagAOX/source/MagAOX
    log_success "Symlinked /opt/MagAOX/source/MagAOX to /vagrant (host folder)"
    usermod -G magaox,magaox-dev vagrant
    log_success "Added vagrant user to magaox,magaox-dev"
else
    if [[ $DIR != /opt/MagAOX/source/MagAOX/setup ]]; then
        if [[ ! -e /opt/MagAOX/source/MagAOX ]]; then
            echo "Cloning new copy of MagAOX codebase"
            git clone $(dirname $DIR) /opt/MagAOX/source/MagAOX
        fi
        cd /opt/MagAOX/source/MagAOX
        git remote remove origin
        git remote add origin https://github.com/magao-x/MagAOX.git
        git fetch
        git branch -u origin/master master
        log_success "In the future, you can re-run this script from /opt/MagAOX/source/MagAOX/setup"
        log_info "(In fact, maybe delete $(dirname $DIR)?)"
    else
        log_info "Running from clone located at $DIR, nothing to do for cloning step"
    fi
fi
# These last steps should work as whatever user is installing, provided
# they are a member of magaox-dev and they have sudo access to install to
# /usr/local. Building as root would leave intermediate build products
# owned by root, which we probably don't want.
#
# On a Vagrant VM, we need to "sudo" to become vagrant since the provisioning
# runs as root.
$MAYBE_SUDO bash -l "$DIR/steps/install_cacao.sh" $TARGET_ENV
$MAYBE_SUDO bash -l "$DIR/steps/install_milkzmq.sh"
$MAYBE_SUDO bash -l "$DIR/steps/install_MagAOX.sh"
