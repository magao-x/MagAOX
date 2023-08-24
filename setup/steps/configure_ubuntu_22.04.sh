#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

log_info "Configuring package source list with Intel oneAPI repositories"
if [[ ! -e /usr/share/keyrings/oneapi-archive-keyring.gpg ]]; then
    mkdir -p /usr/share/keyrings
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null || error_exit "Could not install Intel APT GPG key"
fi
if [[ ! -e /etc/apt/sources.list.d/oneAPI.list ]]; then
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list || error_exit "Could not add Intel APT repository to sources.list.d"
fi
sudo apt update

log_info 'Making /etc/bash.bashrc source /etc/profile.d/*.sh, since graphical sessions appear not to for new Konsoles'
if ! grep -q profile.d /etc/bash.bashrc; then
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

if [[ $MAGAOX_ROLE == AOC ]]; then
  sudo mkdir -p /etc/systemd/logind.conf.d/
  cat <<'HERE' | sudo tee /etc/systemd/logind.conf.d/disable_power_keys.conf
HandlePowerKey=ignore
HandleSuspendKey=ignore
HandleHibernateKey=ignore
HandleRebootKey=ignore
HERE
fi

if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == TOC ]]; then
    log_info "Purging cloud-init"
    sudo NEEDRESTART_SUSPEND=yes apt-get purge -y cloud-init || exit 1
    sudo NEEDRESTART_SUSPEND=yes apt autoremove -y || true

    log_info "Disable waiting for LAN config during boot"
    sudo systemctl mask systemd-networkd-wait-online.service || true
    
    log_info "Ensure UFW firewall is enabled"
    yes | sudo ufw enable || true
    sudo ufw allow ssh || true
    sudo ufw deny http || true
    sudo ufw deny https || true
    sudo ufw allow in from 192.168.0.0/24 || true

    log_info "Use CentOS mountpoint for cpusets"
    sudo mkdir -p /sys/fs/cgroup/cpuset
    cat <<'HERE' | sudo tee /etc/cset.conf || true
mountpoint = /sys/fs/cgroup/cpuset
HERE
fi

log_info "Hush login banners"
sudo touch /etc/skel/.hushlogin

log_info "Disable automatic upgrades"
sudo cp -v /usr/share/unattended-upgrades/20auto-upgrades-disabled /etc/apt/apt.conf.d/

log_info "Done with custom configuration for Ubuntu 22.04"
