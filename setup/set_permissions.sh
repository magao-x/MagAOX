#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

DEFAULT_PASSWORD="extremeAO!"

function creategroup() {
  if [[ ! $(getent group $1) ]]; then
    groupadd $1
    echo "Added group $1"
  else
    echo "Group $1 exists"
  fi
}

function createuser() {
  if getent passwd $1 > /dev/null 2>&1; then
      echo "User account $1 exists"
  else
    useradd $1 -g magaox
    echo -e "$DEFAULT_PASSWORD\n$DEFAULT_PASSWORD" | passwd $1
    echo "Created user account $1 with default password $DEFAULT_PASSWORD"
  fi
}

creategroup magaox
creategroup magaox-dev
createuser xsup

chown -Rv root:root /opt/MagAOX

# Operators should be able to access config and source
# but modifications should be setGID (+s)
chown -Rv :magaox-dev /opt/MagAOX/{config,source}
chown -Rv g+rwXs /opt/MagAOX/{config,source}

# Set logs to writable for non-admin users like xsup
chown -RPv xsup:magaox /opt/MagAOX/logs
# Let operators (not in group magaox) read logs but not write:
chmod -Rv u=rwX,g=rwX,o=rX /opt/MagAOX/logs

# Hide secrets
chmod o-rwx -v /opt/MagAOX/secrets
chmod g-rwx -v /opt/MagAOX/secrets