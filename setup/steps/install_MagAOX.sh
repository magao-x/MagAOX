#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

log_info "Building flatlogs"
cd /opt/MagAOX/source/MagAOX/flatlogs/src
make
make install

log_info "Building MagAOX"
cd /opt/MagAOX/source/MagAOX
make setup
make all
make install

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == vm || $MAGAOX_ROLE == ci ]]; then
    make guis_all
    make guis_install
fi
