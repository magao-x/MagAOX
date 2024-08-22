#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    echo "Becoming root..."
    sudo -H bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

PVCAM_RUNFILE=pvcam_3.10.0.3.run
cd /opt/MagAOX/vendor/teledyne/pvcam || exit 1

chmod +x ./$PVCAM_RUNFILE
if [[ ! -e /opt/pvcam/etc/profile.d/pvcam.sh ]]; then
    echo 'y\nn\n' | bash ./$PVCAM_RUNFILE || exit_with_error "Couldn't install Teledyne Photometrics PVCam for Kinetix"
    log_success "Ran Teledyne Photometrics PVCam installer for Kinetix"
else
    log_info "Existing PVCam install found, use the originall installer or /opt/pvcam/pvcam.uninstall.sh to remove"
fi

PVCAMSDK_RUNFILE=pvcam-sdk_3.10.0.3-1.run
cd /opt/MagAOX/vendor/teledyne/pvcam-sdk || exit 1

chmod +x ./$PVCAMSDK_RUNFILE
if [[ ! -e /opt/pvcam/etc/profile.d/pvcam.sh ]]; then
    echo 'y\nn\n' | bash ./$PVCAMSDK_RUNFILE || exit_with_error "Couldn't install Teledyne Photometrics PVCam for Kinetix"
    log_success "Ran Teledyne Photometrics PVCam installer for Kinetix"
else
    log_info "Existing PVCam install found, use the originall installer or /opt/pvcam/pvcam.uninstall.sh to remove"
fi
