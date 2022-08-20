#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

mkdir -p /opt/MagAOX/vendor/ds9
cd /opt/MagAOX/vendor/ds9

DS9_ARCHIVE=ds9.centos7.8.1.tar.gz
_cached_fetch http://ds9.si.edu/archive/centos7/$DS9_ARCHIVE $DS9_ARCHIVE

tar xf $DS9_ARCHIVE
if [[ ! -e /usr/local/bin/ds9 ]]; then
    sudo ln -s $(realpath ./ds9) /usr/local/bin/ds9
fi