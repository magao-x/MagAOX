#/usr/bin/env bash
sudo mkdir -p /etc/profile.d
echo 'export MAGAOX_ROLE=workstation' | sudo tee /etc/profile.d/magaox.sh
export CI=1
sudo bash -lx ~/MagAOX/setup/steps/ensure_dirs_and_perms.sh
sudo bash -lx ~/MagAOX/setup/install_third_party_deps.sh
sudo shutdown -P now
