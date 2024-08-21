#/usr/bin/env bash
export CI=1
bash -lx ~/MagAOX/setup/provision.sh
sudo shutdown -P now
