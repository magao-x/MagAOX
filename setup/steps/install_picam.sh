#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

PICAM_RUNFILE_CHECKSUM=df337d5ff5bea402669b2283eb534d08
PICAM_RUNFILE=Picam_SDK.run
PICAM_URL=ftp://ftp.piacton.com/Public/Software/Official/PICam/$PICAM_RUNFILE
if [[ ! -e $PICAM_RUNFILE ]]; then
    curl -O $PICAM_URL
fi
if [[ $(md5sum Picam_SDK.run) != "*$PICAM_RUNFILE_CHECKSUM*" ]]; then
    log_error "$PICAM_RUNFILE has md5 checksum $(md5sum Picam_SDK.run)"
    log_error "Expected $PICAM_RUNFILE_CHECKSUM"
    log_error "(Either revise ${BASH_SOURCE[0]} or get the old runfile somewhere)"
    exit 1
fi
exit 1
