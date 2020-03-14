#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
LEVMAR_VERSION="2.6"
#
# LevMar
#
LEVMAR_DIR="./levmar-$LEVMAR_VERSION"
if [[ ! -d $LEVMAR_DIR ]]; then
    _cached_fetch http://users.ics.forth.gr/~lourakis/levmar/levmar-$LEVMAR_VERSION.tgz levmar-$LEVMAR_VERSION.tgz
    tar xzf levmar-$LEVMAR_VERSION.tgz
fi
cd $LEVMAR_DIR
if [[ ! -e /usr/local/lib/liblevmar.a ]]; then
    sed -i 's/#define LINSOLVERS_RETAIN_MEMORY/\/\/#define LINSOLVERS_RETAIN_MEMORY/' levmar.h
    make liblevmar.a
    install liblevmar.a /usr/local/lib/
    install levmar.h /usr/local/include/
fi
