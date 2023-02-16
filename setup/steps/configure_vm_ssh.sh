#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
if [[ "$EUID" == 0 ]]; then
  error_exit "Configure VM ssh as a normal user, not root"
fi
#
# Pre-populate known hosts and hostname aliases for SSH tunneling from the VM
#

if [[ $MAGAOX_ROLE == vm ]]; then
  touch ~/.hushlogin
  mkdir -p $HOME/.ssh
  if [[ ! -e $HOME/.ssh/known_hosts ]]; then
      cat <<'HERE' | sudo tee $HOME/.ssh/known_hosts
rtc ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBFmgoTzcAVYXDZjPFNLfpPz/T/0DQvrXSe9XOly9SD7NcjwN/fRTk+DhrWzdPN5aBsDnnmMS8lFGIcRwnlhUN6o=
icc ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNpRRN65o8TcP2DnkXHdzIqAJ9CAoiz2guLSXjobx7L4meAtphb30nSx5pQqOeysU+otN9PEJH6TWr8KUXBDw6I=
exao1.magao-x.org ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMsOYTn6tlmcatxt1pDfowTtBTsmJ77OMSPl3rNl8+OBKhmpVpX+iBUMKsBDwwVIlqEAa9BfJPbSrpWEWZABv3s=
HERE
  else
      log_info "$HOME/.ssh/known_hosts exists, not overwriting"
  fi
  if [[ ! -e $HOME/.ssh/config ]]; then
    cat << "HERE" | sudo tee $HOME/.ssh/config
IdentityFile $HOME/Home/.ssh/id_ed25519.pub
Host aoc exao1
  HostName exao1.magao-x.org
Host rtc exao2
  HostName rtc
  ProxyJump aoc
Host icc exao3
  HostName icc
  ProxyJump aoc
Host tic exao0
  HostName exao0.as.arizona.edu
Host toc corona
  HostName 192.168.1.62
  ProxyJump exao0
Host *
  User YOURMAGAOXUSERNAME
HERE
  else
      log_info "$HOME/.ssh/config exists, not overwriting"
  fi
  sudo chmod -R u=rwX,g=,o= $HOME/.ssh/
fi
