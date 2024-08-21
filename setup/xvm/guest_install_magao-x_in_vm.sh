#/usr/bin/env bash
sudo rsync -rv ~/MagAOX/ /opt/MagAOX/source/MagAOX/
sudo bash -x /opt/MagAOX/source/MagAOX/setup/steps/ensure_dirs_and_perms.sh
bash -lx /opt/MagAOX/source/MagAOX/setup/steps/install_MagAOX.sh
sudo shutdown -P now
