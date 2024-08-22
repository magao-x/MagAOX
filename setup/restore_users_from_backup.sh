#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh
set -uo pipefail
[ "$#" -eq 1 ] || exit_with_error "Usage: ${BASH_SOURCE[0]} /path/to/folder/of/homes/"
if [[ ! -d $1 ]]; then
    exit_with_error "Specify the path to a folder of backed-up home directories"
fi
allUserFolders=($1/*)
log_warn "Matched these user folders: ${allUserFolders[*]}"
read -p "Ctrl-C to exit, return to continue"

for userFolder in "${allUserFolders[@]}"; do
    userName=$(basename $userFolder)
    log_info "User: $userName"
    if getent passwd $userName > /dev/null 2>&1; then
        log_info "User account $userName exists"
    else
        createuser $userName
    fi
    uid=$(id -u $userName)
    gid=$(id -g $userName)
    sudo rsync -av $userFolder/ /home/$userName/ || exit_with_error "Failed to sync files from $userFolder to /home/$userName/"
    sudo chown -vR $uid:$gid /home/$userName/ || exit_with_error "Failed to normalize ownership to $userName ($(id $userName))"
done
