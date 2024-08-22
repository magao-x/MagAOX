#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    sudo -H bash $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh
set -uo pipefail

cd /home
for username in *; do
    data_path="/data/users/$username"
    if [[ ! -d "$data_path" ]]; then
        mkdir -p "$data_path" || exit 1
        chown "$username:$instrument_group" "$data_path" || exit 1
        chmod g+rxs "$data_path" || exit 1
        log_success "Created $data_path"
    else
        log_info "Data dir $data_path already exists"
    fi
    link_name="/home/$username/data"
    if [[ ! -e "$link_name" ]]; then
        ln -s "$data_path" "$link_name" || exit 1
        log_success "Linked $link_name -> $data_path"
    else
        log_info "Data symlink $link_name already exists"
    fi
done
