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
createuser xdev
chsh -s /sbin/nologin xdev
gpasswd -a xdev magaox-dev

chown -R root:root /opt/MagAOX

# Operators should be able to access config and source
# but modifications should be setGID (+s)
for shared_dir in /opt/MagAOX/config /opt/MagAOX/source; do
  chown -R :magaox-dev $shared_dir
  chmod -R g=rwX $shared_dir
  # n.b. can't be recursive because g+s on files means something else
  # so we find all directories and individually chmod them:
  find /opt/MagAOX/config -type d -exec chmod -v g+s {} \;
done

# Set logs to writable for non-admin users like xsup
chown -RP xsup:magaox /opt/MagAOX/logs
# Let operators (not in group magaox) read logs but not write:
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/logs

# Hide secrets
chmod o-rwx /opt/MagAOX/secrets
chmod g-rwx /opt/MagAOX/secrets