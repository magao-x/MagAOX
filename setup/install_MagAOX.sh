#!/bin/bash
source /opt/rh/devtoolset-7/enable
export PATH="/usr/local/bin:$PATH"
set -exuo pipefail
IFS=$'\n\t'
cd /opt/MagAOX/source/MagAOX/flatlogs/src
make
make install
cd /opt/MagAOX/source/MagAOX
make setup
make all
make install