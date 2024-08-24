#!/usr/bin/env bash
set -o pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh

# Install OS-packaged and a handful of self-built dependencies
# (but not Python environments, proprietary vendor SDKs, or first-party
# dependencies)

# Defines $ID and $VERSION_ID so we can detect which distribution we're on
source /etc/os-release
# Get just the XX beginning of a XX.YY version string
MAJOR_VERSION=${VERSION_ID%.*}

# Install third-party dependencies (including OS-packaged ones) except for vendor SDKs

# Install OS packages first
osPackagesScript="$DIR/steps/install_${ID}_${MAJOR_VERSION}_packages.sh"
sudo -H bash -l $osPackagesScript || exit_with_error "Failed to install packages from $osPackagesScript"

distroSpecificScript="$DIR/steps/configure_${ID}_${MAJOR_VERSION}.sh"
sudo -H bash -l $distroSpecificScript || exit_with_error "Failed to configure ${ID} from $distroSpecificScript"

# Install dependencies for the GUIs
if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC || $MAGAOX_ROLE == ci || $MAGAOX_ROLE == workstation ]]; then
    sudo -H bash -l "$DIR/steps/install_gui_dependencies.sh"
fi

# Install Linux kernel headers
if [[ $MAGAOX_ROLE == ci || $MAGAOX_ROLE == workstation || $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC ]]; then
    if [[ $ID == ubuntu ]]; then
        sudo -i apt install -y linux-headers-generic
    elif [[ $ID == rocky ]]; then
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
sudo -H bash -l "$DIR/steps/install_cfitsio.sh" || exit 1
sudo -H bash -l "$DIR/steps/install_eigen.sh" || exit 1
sudo -H bash -l "$DIR/steps/install_zeromq.sh" || exit 1
sudo -H bash -l "$DIR/steps/install_cppzmq.sh" || exit 1
sudo -H bash -l "$DIR/steps/install_flatbuffers.sh" || exit 1
if [[ $MAGAOX_ROLE == AOC ]]; then
    sudo -H bash -l "$DIR/steps/install_lego.sh"
fi
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == ci ]]; then
    sudo -H bash -l "$DIR/steps/install_basler_pylon.sh"
fi
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == ci ]]; then
    sudo -H bash -l "$DIR/steps/install_edt.sh"
fi

# SuSE packages need either Python 3.6 or 3.10, but Rocky 9.2 has Python 3.9 as /bin/python, so we build our own RPM:
if [[ $ID == rocky && $MAGAOX_ROLE != container ]]; then
  sudo -H bash -l "$DIR/steps/install_cpuset.sh" || exit_with_error "Couldn't install cpuset from source"
fi

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TOC ||  $MAGAOX_ROLE == workstation ]]; then
    # regular old ds9 image viewer
    sudo -H bash -l "$DIR/steps/install_ds9.sh"
fi
