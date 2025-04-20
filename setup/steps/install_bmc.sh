#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILDING_KERNEL_STUFF=1  # disable loading devtoolset-7 for agreement w/ kernel gcc
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/vendor/bmc
bash install.sh
