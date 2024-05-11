#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    sudo -H bash $0 "$@"
    exit $?
fi
set -uo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh


if [[ $# < 1 ]]; then
    log_error "Supply the username for the account to be created as an argument"
    exit 1
fi

createuser $1
