#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd bmc
bash install.sh
echo "export bmc_calib=/opt/MagAOX/calib/dm/bmc_2k" > /etc/profile.d/bmc.sh
