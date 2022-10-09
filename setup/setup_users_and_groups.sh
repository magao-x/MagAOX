#!/bin/bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh

creategroup magaox
creategroup magaox-dev

if [[ $MAGAOX_ROLE != vm ]]; then
  createuser xsup
  createuser guestobs
  sudo passwd --lock guestobs  # SSH login still possible
  creategroup guestobs
  sudo gpasswd -d guestobs magaox || true  # prevent access for shenanigans
  sudo gpasswd -a guestobs guestobs || true
  sudo chown guestobs:guestobs /data/users/guestobs
  sudo chmod g+rwX /data/users/guestobs
  if [[ -z $(groups | tr ' ' '\n' | grep 'guestobs$') ]]; then
    sudo gpasswd -a xsup guestobs
    log_success "Added xsup to group guestobs"
  fi

  if sudo test ! -e /home/xsup/.ssh/id_ed25519; then
    $REAL_SUDO -u xsup ssh-keygen -t ed25519 -N "" -f /home/xsup/.ssh/id_ed25519 -q
  fi
  if ! grep -q magaox-dev /etc/pam.d/su; then
    cat <<'HERE' | sudo sed -i '/pam_rootok.so$/r /dev/stdin' /etc/pam.d/su
auth            [success=ignore default=1] pam_succeed_if.so user = xsup
auth            sufficient      pam_succeed_if.so use_uid user ingroup magaox-dev
HERE
    log_info "Modified /etc/pam.d/su"
  else
    log_info "/etc/pam.d/su already includes reference to magaox-dev, not modifying"
  fi
fi
if [[ $EUID != 0 ]]; then
  if [[ -z $(groups | tr ' ' '\n' | grep 'magaox-dev$') ]]; then
    sudo gpasswd -a $USER magaox-dev
    log_success "Added $USER to group magaox-dev"
    log_warn "Note: You will need to log out and back in before this group takes effect"
  fi
  if [[ -z $(groups | tr ' ' '\n' | grep 'magaox$') ]]; then
    sudo gpasswd -a $USER magaox
    log_success "Added $USER to group magaox"
    log_warn "Note: You will need to log out and back in before this group takes effect"
  fi
fi
