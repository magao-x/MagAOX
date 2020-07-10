#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
#
# Pre-populate known hosts and hostname aliases for SSH tunneling from the VM
#
if [[ $MAGAOX_ROLE == vm ]]; then
  mkdir -p /vagrant/vm/ssh
  cat <<'HERE' | sudo tee /vagrant/vm/ssh/config
IdentityFile /vagrant/vm/ssh/magaox_ssh_key
Host aoc
  HostName exao1.magao-x.org
Host rtc
  HostName rtc
  ProxyJump aoc
Host icc
  HostName icc
  ProxyJump aoc
Host *
  User YOURUSERNAME
HERE
  mkdir -p /home/vagrant/.ssh
  if [[ ! -e /home/vagrant/.ssh/known_hosts ]]; then
      cat <<'HERE' | sudo tee /home/vagrant/.ssh/known_hosts
rtc ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBFmgoTzcAVYXDZjPFNLfpPz/T/0DQvrXSe9XOly9SD7NcjwN/fRTk+DhrWzdPN5aBsDnnmMS8lFGIcRwnlhUN6o=
icc ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNpRRN65o8TcP2DnkXHdzIqAJ9CAoiz2guLSXjobx7L4meAtphb30nSx5pQqOeysU+otN9PEJH6TWr8KUXBDw6I=
exao1.magao-x.org ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMsOYTn6tlmcatxt1pDfowTtBTsmJ77OMSPl3rNl8+OBKhmpVpX+iBUMKsBDwwVIlqEAa9BfJPbSrpWEWZABv3s=
HERE
  else
      log_info "/home/vagrant/.ssh/known_hosts exists, not overwriting"
  fi
  if [[ ! -e /home/vagrant/.ssh/config ]]; then
    cat <<'HERE' | sudo tee /home/vagrant/.ssh/config
Include /vagrant/vm/ssh/config
HERE
  else
      log_info "/home/vagrant/.ssh/config exists, not overwriting"
  fi
  sudo chown -R vagrant:vagrant /home/vagrant/.ssh/
  sudo chmod -R u=rwX,g=,o= /home/vagrant/.ssh/
fi
