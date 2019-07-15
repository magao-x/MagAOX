#!/bin/bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DEFAULT_PASSWORD="extremeAO!"

function creategroup() {
  if [[ ! $(getent group $1) ]]; then
    /bin/sudo groupadd $1
    echo "Added group $1"
  else
    echo "Group $1 exists"
  fi
}

function createuser() {
  if getent passwd $1 > /dev/null 2>&1; then
      echo "User account $1 exists"
  else
    /bin/sudo useradd $1 -g magaox
    echo -e "$DEFAULT_PASSWORD\n$DEFAULT_PASSWORD" | passwd $1
    echo "Created user account $1 with default password $DEFAULT_PASSWORD"
  fi
}

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
  echo "Installed new /etc/pam.d/su"
else
  echo "/etc/pam.d/su already includes reference to magaox-dev, not overwriting"
fi
if [[ $EUID != 0 ]]; then
  if [[ -z $(groups | grep magaox-dev) ]]; then
    /bin/sudo gpasswd -a $USER magaox-dev
    echo "Added $USER to group magaox-dev"
    echo "Note: You will need to log out and back in before this group takes effect"
  fi
fi
