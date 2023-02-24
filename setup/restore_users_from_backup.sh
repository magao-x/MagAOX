#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh
if [[ -z $1 ]]; then
    error_exit "Specify the path to a folder of backed-up home directories"
fi
for username in $1/*; do
    log_info "User: $(basename $username)"
done