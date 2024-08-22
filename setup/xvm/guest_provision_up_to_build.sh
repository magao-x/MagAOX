#/usr/bin/env bash
function shutdownVM() {
    echo 'Shutting down VM from within guest...'
    sudo shutdown -P now
}
trap shutdownVM EXIT
set -x
export CI=1
export _skip3rdPartyDeps=1
bash -lx ~/MagAOX/setup/provision.sh || exit 1
