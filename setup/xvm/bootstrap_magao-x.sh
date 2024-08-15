#/usr/bin/env bash
# Install MagAO-X software on a fresh Rocky Linux installation.
git clone https://github.com/magao-x/MagAOX.git
cd MagAOX/setup/
echo 'export MAGAOX_ROLE=workstation' | sudo tee /etc/profile.d/magaox.sh
export CI=1
bash -lx provision.sh
sudo shutdown -P now
