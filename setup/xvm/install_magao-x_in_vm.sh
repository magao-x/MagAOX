#/usr/bin/env bash
sudo rsync -rv ~/MagAOX/ /opt/MagAOX/source/MagAOX/
bash -lx /opt/MagAOX/source/MagAOX/setup/steps/install_MagAOX.sh
sudo shutdown -P now
