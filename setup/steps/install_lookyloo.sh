#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

commit_ish=main
orgname=magao-x
reponame=lookyloo
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir || exit 1
git checkout $commit_ish || exit 1

cd $parentdir/$reponame || exit 1
sudo -H /opt/conda/bin/pip install -e . || exit_with_error "Could not pip install $reponame"
lookyloo -h 2>&1 > /dev/null || exit_with_error "'lookyloo -h' command exited with an error, or was not found"
UNIT_PATH=/etc/systemd/system/
if [[ $MAGAOX_ROLE == AOC ]]; then
    sudo cp $DIR/../systemd_units/lookyloo.service $UNIT_PATH/lookyloo.service || exit 1
    log_success "Installed lookyloo.service to $UNIT_PATH"

    sudo -H systemctl daemon-reload || exit 1
    sudo systemctl enable lookyloo || exit 1
    log_success "Enabled lookyloo service"
    sudo systemctl start lookyloo || exit 1
    log_success "Started lookyloo service"
fi
