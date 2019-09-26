#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

PACKAGE_VERSION=6.1.4
PACKAGE_FILENAME=qwt-${PACKAGE_VERSION}.tar.bz2
QWT_ROOT_DEFAULT=/usr/local/qwt-6.1.4

cd /opt/MagAOX/vendor
if [[ $ID == centos && $VERSION_ID == 7 ]]; then
    _cached_fetch https://downloads.sourceforge.net/project/qwt/qwt/${PACKAGE_VERSION}/${PACKAGE_FILENAME} ${PACKAGE_FILENAME}
    tar xvf ${PACKAGE_FILENAME}
    cd qwt-${PACKAGE_VERSION}/
    sed -E -i 's/#?QWT_CONFIG     += QwtSvg/#QWT_CONFIG     += QwtSvg/g' qwtconfig.pri
    qmake qwt.pro
    make -j 32
    sudo make install
    qmake -set QMAKEFEATURES ${QWT_ROOT_DEFAULT}/features
fi

