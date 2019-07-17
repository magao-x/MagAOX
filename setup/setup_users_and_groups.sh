#!/bin/bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh

creategroup magaox
creategroup magaox-dev
createuser xsup
if grep -vq magaox-dev /etc/pam.d/su; then
  sudo tee /etc/pam.d/su >/dev/null <<EOF
#%PAM-1.0
auth            sufficient      pam_rootok.so
auth            [success=ignore default=1] pam_succeed_if.so user = xsup
auth            sufficient      pam_succeed_if.so use_uid user ingroup magaox-dev
# Uncomment the following line to implicitly trust users in the "wheel" group.
#auth           sufficient      pam_wheel.so trust use_uid
# Uncomment the following line to require a user to be in the "wheel" group.
#auth           required        pam_wheel.so use_uid
auth            substack        system-auth
auth            include         postlogin
account         sufficient      pam_succeed_if.so uid = 0 use_uid quiet
account         include         system-auth
password        include         system-auth
session         include         system-auth
session         include         postlogin
session         optional        pam_xauth.so
EOF
  log_info "Installed new /etc/pam.d/su"
else
  log_info "/etc/pam.d/su already includes reference to magaox-dev, not overwriting"
fi
if [[ $EUID != 0 ]]; then
  if [[ -z $(groups | grep magaox-dev) ]]; then
    /bin/sudo gpasswd -a $USER magaox-dev
    log_success "Added $USER to group magaox-dev"
    log_warn "Note: You will need to log out and back in before this group takes effect"
  fi
fi
