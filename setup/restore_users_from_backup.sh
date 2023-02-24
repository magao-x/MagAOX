#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh
[ "$#" -eq 1 ] || error_exit "Usage: ${BASH_SOURCE[0]} /path/to/folder/of/homes/"
if [[ ! -d $1 ]]; then
    error_exit "Specify the path to a folder of backed-up home directories"
fi
allUserFolders=($1/*)
log_warn "Matched these user folders: $allUserFolders"
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
    echo sudo rsync -av $userFolder/ /home/$userName/ || error_exit "Failed to sync files from $userFolder to /home/$userName/"
    echo sudo chown -vR $uid:$gid /home/$userName/ || error_exit "Failed to normalize ownership to $userName ($(id $userName))"
done
