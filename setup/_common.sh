#!/bin/bash
function log_error() {
    echo -e "$(tput setaf 1)$1$(tput sgr0)"
}
function log_success() {
    echo -e "$(tput setaf 2)$1$(tput sgr0)"
}
function log_warn() {
    echo -e "$(tput setaf 3)$1$(tput sgr0)"
}
function log_info() {
    echo -e "$(tput setaf 4)$1$(tput sgr0)"
}

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
    /bin/sudo mkdir -p /home/$1/.ssh
    /bin/sudo touch /home/$1/.ssh/authorized_keys
    /bin/sudo chown -R $1:magaox /home/$1/.ssh
    /bin/sudo chmod -R u=rwx,g=,o= /home/$1/.ssh
    /bin/sudo chmod u=rw,g=,o= /home/$1/.ssh/authorized_keys
    log_success "Created user account $1 with default password $DEFAULT_PASSWORD"
    log_info "Append an ecdsa or ed25519 key to /home/$1/.ssh/authorized_keys to enable SSH login"
  fi
}
# We work around the buggy devtoolset /bin/sudo wrapper in provision.sh, but
# that means we have to explicitly enable it ourselves.
# (This crap again: https://bugzilla.redhat.com/show_bug.cgi?id=1319936)
if [[ -e /opt/rh/devtoolset-7/enable ]]; then
    source /opt/rh/devtoolset-7/enable
fi
# root doesn't get /usr/local/bin on their path, so add it
# (why? https://serverfault.com/a/838552)
if [[ $PATH != "*/usr/local/bin*" ]]; then
    export PATH="/usr/local/bin:$PATH"
fi
