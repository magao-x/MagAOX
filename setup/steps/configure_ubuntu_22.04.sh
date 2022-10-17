#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == TOC ]]; then
    log_info "Purging cloud-init"
    sudo apt-get purge -y cloud-init || exit 1
    sudo apt autoremove -y || true
fi

log_info "Disable waiting for LAN config during boot"
systemctl mask systemd-networkd-wait-online.service || true


log_info "Configuring package source list with Intel oneAPI repositories"
if [[ ! -e /usr/share/keyrings/oneapi-archive-keyring.gpg ]]; then
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null || exit
fi
if [[ ! -e /etc/apt/sources.list.d/oneAPI.list ]]; then
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list || exit
fi
sudo apt update

log_info 'Making /etc/bash.bashrc source /etc/profile.d/*.sh, since graphical sessions appear not to for new Konsoles'
if ! grep -q bashrc.d /etc/bash.bashrc; then
cat <<'HERE' | sudo tee -a /etc/bash.bashrc
if [ -d /etc/profile.d ]; then
  for i in /etc/profile.d/*.sh; do
    if [ -r $i ]; then
      . $i
    fi
  done
  unset i
fi
HERE
fi

yes | sudo ufw enable || exit 1
sudo ufw allow 22 || exit 1

log_info "Done with custom configuration for Ubuntu 22.04"
