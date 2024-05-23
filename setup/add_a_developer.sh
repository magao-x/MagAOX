#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    sudo -H bash $0 "$@"
    exit $?
fi
set -uo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh


$DIR/add_a_user.sh $1
gpasswd -a $1 magaox-dev
if [[ ! $(getent group $1) ]]; then
    sudo groupadd $1
    echo "Added group $1"
else
    echo "Group $1 exists"
fi
source /etc/os-release
if [[ $ID == ubuntu ]]; then
    admins_group=sudo
else
    admins_group=wheel
fi
gpasswd -a $1 $admins_group
log_success "Added $1 to groups magaox-dev and $admins_group"
