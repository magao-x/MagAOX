#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
PYLON_VERSION="5.2.0.13457"
#
# Basler camera Pylon framework
#
PYLON_DIR="pylon-$PYLON_VERSION-x86_64"
if [[ ! -d $PYLON_DIR ]]; then
    _cached_fetch https://www.baslerweb.com/fp-1551786516/media/downloads/software/pylon_software/$PYLON_DIR.tar.gz $PYLON_DIR.tar.gz
    tar xzf $PYLON_DIR.tar.gz || exit 1
fi
cd $PYLON_DIR || exit 1
tar -C /opt -xzf pylonSDK*.tar.gz || exit 1
# Replacement for the important parts of setup-usb.sh
BASLER_RULES_FILE=69-basler-cameras.rules
UDEV_RULES_DIR=/etc/udev/rules.d
if [[ ! -e $UDEV_RULES_DIR/$BASLER_RULES_FILE ]]; then
    cp $BASLER_RULES_FILE $UDEV_RULES_DIR || exit 1
fi
