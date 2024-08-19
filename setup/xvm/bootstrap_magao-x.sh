#/usr/bin/env bash
sudo mkdir -p /etc/profile.d
echo 'export MAGAOX_ROLE=workstation' | sudo tee /etc/profile.d/magaox.sh
export CI=1
bash -lx /opt/MagAOX/source/MagAOX/provision.sh
sudo shutdown -P now
