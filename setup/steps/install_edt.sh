#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
# lspci needed to install EDT framegrabber driver
yum install -y pciutils
EDT_RPM_FILENAME=EDTpdv-5.5.7-2.noarch.rpm
if [[ ! -e $EDT_RPM_FILENAME ]]; then
    curl -OL https://edt.com/wp-content/uploads/2019/05/$EDT_RPM_FILENAME
fi
yum install -y $EDT_RPM_FILENAME || true
