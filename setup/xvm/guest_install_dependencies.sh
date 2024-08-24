#/usr/bin/env bash
function shutdownVM() {
    echo 'Shutting down VM from within guest...'
    sudo shutdown -P now
}
trap shutdownVM EXIT
set -x
sudo mkdir -p /etc/profile.d || exit 1
echo 'export MAGAOX_ROLE=workstation' | sudo tee /etc/profile.d/magaox.sh || exit 1
export CI=1
sudo bash -lx ~/MagAOX/setup/steps/ensure_dirs_and_perms.sh || (echo 'Failed to create dirs' && exit 1)
sudo bash -lx ~/MagAOX/setup/steps/install_fftw.sh || (echo 'Failed to install FFTW' && exit 1)
echo 'Installed third-party dependencies'
