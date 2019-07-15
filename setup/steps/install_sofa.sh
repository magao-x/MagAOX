#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
SOFA_REV="2018_0130_C"
SOFA_REV_DATE=$(echo $SOFA_REV | tr -d _C)
#
# SOFA
#
if [[ ! -d ./sofa ]]; then
    if [[ ! -e sofa_c-$SOFA_REV_DATE.tar.gz ]]; then
        curl -OL http://www.iausofa.org/$SOFA_REV/sofa_c-$SOFA_REV_DATE.tar.gz
    fi
    tar xzf sofa_c-$SOFA_REV_DATE.tar.gz
    echo "Downloaded and unpacked 'sofa' from sofa_c_-$SOFA_REV_DATE.tar.gz"
fi
cd sofa/$SOFA_REV_DATE/c/src
if [[ ! -e /usr/local/lib/libsofa_c.a ]]; then
    make "CFLAGX=-pedantic -Wall -W -O -fPIC" "CFLAGF=-c -pedantic -Wall -W -O -fPIC"
    make install INSTALL_DIR=/usr/local
fi
