#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

mkdir -p /opt/MagAOX/vendor/ds9 || exit 1
cd /opt/MagAOX/vendor/ds9 || exit 1

DS9_ARCHIVE=ds9.centos7.8.1.tar.gz
_cached_fetch http://ds9.si.edu/archive/centos7/$DS9_ARCHIVE $DS9_ARCHIVE || exit 1

tar xf $DS9_ARCHIVE || exit 1
if [[ ! -e /usr/local/bin/ds9 ]]; then
    sudo ln -s $(realpath ./ds9) /usr/local/bin/ds9 || exit 1
fi