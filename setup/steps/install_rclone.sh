#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
cd /opt/MagAOX/vendor
PACKAGE_VERSION=1.60.0
arch=$(uname -p)
PACKAGE_DIR=rclone-v${PACKAGE_VERSION}-linux
if [[ $arch == "x86_64" ]]; then
    PACKAGE_DIR="$PACKAGE_DIR-amd64"
elif [[ $arch == "aarch64" ]]; then
    PACKAGE_DIR="$PACKAGE_DIR-arm64"
else
    exit_with_error "Unknown arch: $arch"
fi
PACKAGE_ARCHIVE=$PACKAGE_DIR.zip
if [[ ! -d $PACKAGE_DIR ]]; then
    _cached_fetch https://downloads.rclone.org/v${PACKAGE_VERSION}/$PACKAGE_ARCHIVE $PACKAGE_ARCHIVE || exit 1
    unzip $PACKAGE_ARCHIVE || exit 1
fi
cd $PACKAGE_DIR || exit 1
sudo install ./rclone /usr/local/bin || exit 1
sudo mkdir -p /usr/local/share/man/man1/ || exit 1
sudo install ./rclone.1 /usr/local/share/man/man1/ || exit 1
sudo ln -sf /usr/local/bin/rclone /sbin/mount.rclone || exit 1
