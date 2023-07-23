#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

commit_ish=main
orgname=magao-x
reponame=lookyloo
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir
git checkout $commit_ish

cd $parentdir/$reponame
pip install -e . || exit_error "Could not pip install $reponame"
lookyloo -h 2>&1 > /dev/null || exit_error "'lookyloo -h' command exited with an error, or was not found"

if [[ $MAGAOX_ROLE == AOC ]]; then
    cp $DIR/../systemd_units/lookyloo.service $UNIT_PATH/lookyloo.service
    log_success "Installed lookyloo.service to $UNIT_PATH"

    systemctl daemon-reload
    systemctl enable lookyloo
    log_success "Enabled lookyloo service"
    systemctl start lookyloo
    log_success "Started lookyloo service"
fi
