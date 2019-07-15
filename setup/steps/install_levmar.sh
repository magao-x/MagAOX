#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -ue
LEVMAR_VERSION="2.6"
#
# LevMar
#
LEVMAR_DIR="./levmar-$LEVMAR_VERSION"
if [[ ! -d $LEVMAR_DIR ]]; then
    if [[ ! -e levmar-$LEVMAR_VERSION.tgz ]]; then
        curl -OLA "Mozilla/5.0" http://users.ics.forth.gr/~lourakis/levmar/levmar-$LEVMAR_VERSION.tgz
    fi
    tar xzf levmar-$LEVMAR_VERSION.tgz
fi
cd $LEVMAR_DIR
if [[ ! -e /usr/local/lib/liblevmar.a ]]; then
    make liblevmar.a
    install liblevmar.a /usr/local/lib/
fi
