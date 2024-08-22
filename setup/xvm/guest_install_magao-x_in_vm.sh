#/usr/bin/env bash
function shutdownVM() {
    echo 'Shutting down VM from within guest...'
    sudo shutdown -P now
}
trap shutdownVM EXIT
set -x
sudo rsync -rv ~/MagAOX/ /opt/MagAOX/source/MagAOX/ || exit 1
sudo bash -x /opt/MagAOX/source/MagAOX/setup/steps/ensure_dirs_and_perms.sh || exit 1
bash -lx /opt/MagAOX/source/MagAOX/setup/steps/install_MagAOX.sh || exit 1
