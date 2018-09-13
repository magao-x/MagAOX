#!/bin/bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IFS=$'\n\t'

envswitch=${1:---prod}
if [[ "$envswitch" == "--dev" ]]; then
  ENV=dev
elif [[ "$envswitch" == "--prod" ]]; then
  ENV=prod
else
  cat <<'HERE'
Usage: setup_users_and_groups.sh [--dev] [--prod]
Automate installation of from-package-manager and from-source
software dependencies for MagAO-X

  --prod  (default) Set up for production (add current user to magaox-dev)
  --dev   Set up for local development (don't bother)
HERE
  exit 1
fi

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
  /bin/sudo cp -v "$DIR/setup/enable_su_xsup_for_magaox-dev" /etc/pam.d/su
  echo "Installed new /etc/pam.d/su"
else
  echo "/etc/pam.d/su already includes reference to magaox-dev, not overwriting"
fi
if [[ $ENV == "prod" ]]; then
  if [[ -z $(groups | grep magaox-dev) ]]; then
    /bin/sudo gpasswd -a $USER magaox-dev
    echo "Added $USER to group magaox-dev"
    echo "Note: You will need to log out and back in before this group takes effect"
  fi
fi