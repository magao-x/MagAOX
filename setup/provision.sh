#!/bin/bash
set -eo pipefail
if [[ -d /vagrant ]]; then
    TARGET_ENV=vagrant
    DIR="/vagrant/setup"
    echo "Setting up for VM use"
    /bin/sudo bash -l "$DIR/setup_users_and_groups.sh"
    /bin/sudo yum install -y kernel-devel
else
    TARGET_ENV=instrument
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
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
    /bin/sudo -v
    # Keep the sudo timestamp updated until this script exits
    while true; do /bin/sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
fi
source $DIR/_common.sh

VENDOR_SOFTWARE_BUNDLE=$DIR/vendor_software.tar.gz
# if [[ ! -e $VENDOR_SOFTWARE_BUNDLE ]]; then
#     echo "Couldn't find vendor software bundle at location $VENDOR_SOFTWARE_BUNDLE"
#     echo "(Generate with ~/Box/MagAO-X/Vendor\ Software/generate_bundle.sh)"
#     if [[ $TARGET_ENV == "instrument" ]]; then
#         exit 1
#     fi
# fi
# die on uninitialized variables (typo guard)
set -u

## Set up file structure and permissions
/bin/sudo bash -l "$DIR/steps/make_directories.sh" $TARGET_ENV

## Build third-party dependencies under /opt/MagAOX/vendor
cd /opt/MagAOX/vendor
/bin/sudo bash -l "$DIR/steps/install_os_packages.sh"
/bin/sudo bash -l "$DIR/steps/install_devtoolset-7.sh"
/bin/sudo bash -l "$DIR/steps/install_mkl_tarball.sh"
/bin/sudo bash -l "$DIR/steps/install_cuda.sh"
/bin/sudo bash -l "$DIR/steps/install_fftw.sh"
/bin/sudo bash -l "$DIR/steps/install_cfitsio.sh"
/bin/sudo bash -l "$DIR/steps/install_sofa.sh"
/bin/sudo bash -l "$DIR/steps/install_eigen.sh"
/bin/sudo bash -l "$DIR/steps/install_levmar.sh"
/bin/sudo bash -l "$DIR/steps/install_flatbuffers.sh"
/bin/sudo bash -l "$DIR/steps/install_xrif.sh"
/bin/sudo bash -l "$DIR/steps/install_magma.sh"
/bin/sudo bash -l "$DIR/steps/install_basler_pylon.sh"
/bin/sudo bash -l "$DIR/steps/install_edt.sh"
/bin/sudo bash -l "$DIR/steps/install_picam.sh"

## Install proprietary / non-public software
if [[ -e $VENDOR_SOFTWARE_BUNDLE ]]; then
    # Extract bundle
    BUNDLE_TMPDIR=/tmp/vendor_software_bundle
    mkdir -p /tmp/vendor_software_bundle
    tar xzf $VENDOR_SOFTWARE_BUNDLE -C $BUNDLE_TMPDIR
    for vendorname in alpao bmc; do
        if [[ ! -d /opt/MagAOX/vendor/$vendorname ]]; then
            cp -R $BUNDLE_TMPDIR/$vendorname /opt/MagAOX/vendor
        else
            echo "/opt/MagAOX/vendor/$vendorname exists, not overwriting files"
            echo "(but they're in $BUNDLE_TMPDIR/$vendorname if you want them)"
        fi
    done
    /bin/sudo bash -l "$DIR/steps/install_alpao.sh"
    /bin/sudo bash -l "$DIR/steps/install_bmc.sh"
fi

## Build first-party dependencies
cd /opt/MagAOX/source
/bin/sudo bash -l "$DIR/steps/install_mxlib.sh"
source /etc/profile.d/mxmakefile.sh

## Verify all permissions are set correctly
/bin/sudo bash -l "$DIR/steps/set_permissions.sh"


## Build MagAO-X and install sources to /opt/MagAOX/source/MagAOX
/bin/sudo bash -l "$DIR/steps/install_MagAOX_osdeps.sh"

if [[ "$TARGET_ENV" == "vagrant" ]]; then
    # Create or replace symlink to sources so we develop on the host machine's copy
    # (unlike prod, where we install a new clone of the repo to this location)
    ln -nfs /vagrant /opt/MagAOX/source/MagAOX
    usermod -G magaox,magaox-dev vagrant
    /bin/sudo -u vagrant bash "$DIR/steps/install_MagAOX.sh"
    echo "Finished!"
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
        echo "In the future, you can re-run this script from /opt/MagAOX/source/MagAOX/setup"
        echo "(In fact, maybe delete $(dirname $DIR)?)"
    else
        echo "Running from clone located at $DIR, nothing to do for cloning step"
    fi
    # The last step should work as whatever user is installing, provided
    # they are a member of magaox-dev and they have sudo access to install to
    # /usr/local. Building as root would leave intermediate build products
    # owned by root, which we probably don't want.
    bash "$DIR/steps/install_MagAOX.sh"
    echo "Finished!"
fi
