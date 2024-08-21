#/usr/bin/env bash
sudo mkdir -p /etc/profile.d
echo 'export MAGAOX_ROLE=workstation' | sudo tee /etc/profile.d/magaox.sh
export CI=1
sudo bash -lx ~/MagAOX/setup/steps/install_rocky_9_packages.sh
sudo bash -lx ~/MagAOX/setup/steps/configure_rocky_9.sh
sudo bash -lx ~/MagAOX/setup/steps/install_gui_dependencies.sh
sudo shutdown -P now
