#!/bin/bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh

creategroup magaox
creategroup magaox-dev
createuser xsup
if ! grep -q magaox-dev /etc/pam.d/su; then
  cat <<'HERE' | sudo sed -i '/pam_rootok.so$/r /dev/stdin' /etc/pam.d/su
auth            [success=ignore default=1] pam_succeed_if.so user = xsup
auth            sufficient      pam_succeed_if.so use_uid user ingroup magaox-dev
HERE
  log_info "Modified /etc/pam.d/su"
else
  log_info "/etc/pam.d/su already includes reference to magaox-dev, not modifying"
fi
if [[ $EUID != 0 ]]; then
  if [[ -z $(groups | grep magaox-dev) ]]; then
    /bin/sudo gpasswd -a $USER magaox-dev
    log_success "Added $USER to group magaox-dev"
    log_warn "Note: You will need to log out and back in before this group takes effect"
  fi
fi
