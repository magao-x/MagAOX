#/usr/bin/env bash
# Install MagAO-X software on a fresh Rocky Linux 9.3 installation.
# This script is intended to be run on a fresh Rocky Linux 9.3 installation.
# It will install the MagAO-X software and its dependencies.
git clone https://github.com/magao-x/MagAOX.git
cd MagAOX/setup/
echo 'export MAGAOX_ROLE=workstation' | sudo tee /etc/profile.d/magaox.sh
bash -lx provision.sh
sudo shutdown -P now