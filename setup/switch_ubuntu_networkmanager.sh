#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh
set -uo pipefail

cat <<'HERE' | sudo tee /etc/network/interfaces
auto lo
iface lo inet loopback
HERE

sudo systemctl disable systemd-networkd
sudo systemctl mask systemd-networkd
sudo systemctl stop systemd-networkd

sudo -i apt install -y network-manager

cat <<'HERE' | sudo tee /etc/netplan/00-installer-config.yaml || exit 1
network:
  version: 2
  renderer: NetworkManager
HERE

sudo netplan generate
sudo systemctl unmask NetworkManager
sudo systemctl enable NetworkManager
sudo systemctl start NetworkManager || exit 1
nmcli | cat || exit 1

log_info "Might be a good idea to restart"