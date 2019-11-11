#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

PICAM_RUNFILE_CHECKSUM=37bedee6c828e364750e9b5fd7c2a420
PICAM_RUNFILE=Picam_SDK-v5.7.2.run
PICAM_URL=ftp://ftp.piacton.com/Public/Software/Official/PICam/Archives/$PICAM_RUNFILE
_cached_fetch $PICAM_URL $PICAM_RUNFILE
if [[ $(md5sum ./$PICAM_RUNFILE) != *$PICAM_RUNFILE_CHECKSUM* ]]; then
    log_error "$PICAM_RUNFILE has md5 checksum $(md5sum ./$PICAM_RUNFILE)"
    log_error "Expected $PICAM_RUNFILE_CHECKSUM"
    log_error "(Either revise ${BASH_SOURCE[0]} or get the old runfile somewhere)"
    exit 1
fi
chmod +x ./$PICAM_RUNFILE
if [[ ! -e /opt/PrincetonInstruments/picam ]]; then
    # This (|| true) isn't great, but sometimes it has a nonzero
    # exit code and sometimes it doesn't. Both states are accompanied
    # by "Picam v5.7.2 Installation complete."
    yes yes | ./$PICAM_RUNFILE || true
    log_success "Ran Picam SDK installer"
fi
chmod a+rX -R /opt/pleora
chmod a+rX -R /opt/PrincetonInstruments
chmod g+xr,o+xr /usr/local/lib/libftd2xx.so.1.4.6
echo "if [[ \"\$EUID\" != 0 ]]; then source /opt/pleora/ebus_sdk/x86_64/bin/set_puregev_env; fi" > /etc/profile.d/picam_pleora_env.sh
log_success "Princeton Instruments Picam SDK installed"
