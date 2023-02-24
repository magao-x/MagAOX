#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh
if [[ -z $1 ]]; then
    error_exit "Specify the path to a folder of backed-up home directories"
fi
homePrefix=$1
for userFolder in $homePrefix/*; do
    userName=$(basename $userFolder)
    if [[ $userName == '*' ]]; then
        error_exit "No folders in $homePrefix to match"
    fi
    log_info "User: $userName"
    if getent passwd $userName > /dev/null 2>&1; then
        log_info "User account $userName exists"
    else
        createuser $userName
    fi
    uid=$(id -u $userName)
    gid=$(id -g $userName)
    sudo rsync -av $userFolder/ /home/$userName/
    sudo chown -vR $uid:$gid /home/$userName/
done