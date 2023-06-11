#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    echo "Becoming root..."
    sudo bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

PVCAM_RUNFILE=pvcam_3.9.12.11.run
cd /opt/MagAOX/vendor/teledyne/pvcam

chmod +x ./$PVCAM_RUNFILE
if [[ ! -e /opt/pvcam/etc/profile.d/pvcam.sh ]]; then
    echo 'y\nn\n' | bash ./$PVCAM_RUNFILE || error_exit "Couldn't install Teledyne Photometrics PVCam for Kinetix"
    log_success "Ran Teledyne Photometrics PVCam installer for Kinetix"
else
    log_info "Existing PVCam install found, use the originall installer or /opt/pvcam/pvcam.uninstall.sh to remove"
fi
# if [[ ! -e /opt/PrincetonInstruments/picam ]]; then
#     # This (|| true) isn't great, but sometimes it has a nonzero
#     # exit code and sometimes it doesn't. Both states are accompanied
#     # by "Picam v5.7.2 Installation complete."
#     yes yes | ./$PICAM_RUNFILE || true
#     log_success "Ran Picam SDK installer"
# fi
# if [[ ! -e /opt/PrincetonInstruments/picam ]]; then
#     exit_error "Installer failed to create /opt/PrincetonInstruments/picam, aborting"
# fi
# chmod a+rX -R /opt/pleora
# chmod a+rX -R /opt/PrincetonInstruments
# chmod g+xr,o+xr /usr/local/lib/libftd2xx.so.1.4.6
# echo "if [[ \"\$EUID\" != 0 ]]; then source /opt/pleora/ebus_sdk/x86_64/bin/set_puregev_env; fi" > /etc/profile.d/picam_pleora_env.sh
# log_success "Princeton Instruments Picam SDK installed"
