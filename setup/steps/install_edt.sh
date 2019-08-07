#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

EDT_FILENAME=EDTpdv_lnx_5.5.8.2.run
if [[ ! -e $EDT_FILENAME ]]; then
    curl -L https://edt.com/downloads/pdv_5-5-8-2_run/ > $EDT_FILENAME
fi
chmod +x $EDT_FILENAME
./$EDT_FILENAME -- --default
