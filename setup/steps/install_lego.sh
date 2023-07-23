#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -o pipefail
LEGO_VERSION=v4.9.0
LEGO_ARCHIVE=lego_${LEGO_VERSION}_linux_amd64.tar.gz
LEGO_FOLDER=/opt/MagAOX/vendor/lego_$LEGO_VERSION
mkdir -p $LEGO_FOLDER
cd $LEGO_FOLDER
if [[ ! -e lego ]]; then
    _cached_fetch https://github.com/go-acme/lego/releases/download/v4.9.0/$LEGO_ARCHIVE $LEGO_ARCHIVE
    tar xf $LEGO_ARCHIVE
fi
sudo ln -sfv $(realpath ./lego) /usr/local/bin/lego
sudo mkdir -p /opt/lego
sudo chown :$instrument_dev_group /opt/lego
sudo chmod -R u=rwX,g=rwX,o=x /opt/lego
setgid_all /opt/lego