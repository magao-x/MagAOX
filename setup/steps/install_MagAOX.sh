#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

log_info "Building flatlogs"
cd /opt/MagAOX/source/MagAOX/flatlogs/src
make clean
make
make install

log_info "Building MagAOX"
cd /opt/MagAOX/source/MagAOX
make setup
make all_clean
make all
make install
