#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail
if [[ "$EUID" == 0 ]]; then
  exit_with_error "Configure VM ssh as a normal user, not root"
fi
#
# Pre-populate known hosts and hostname aliases for SSH tunneling from the VM
#
touch ~/.hushlogin || exit 1
mkdir -p $HOME/.ssh || exit 1
if [[ ! -e $HOME/.ssh/known_hosts ]]; then
    cat <<'HERE' | tee $HOME/.ssh/known_hosts
rtc ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBFmgoTzcAVYXDZjPFNLfpPz/T/0DQvrXSe9XOly9SD7NcjwN/fRTk+DhrWzdPN5aBsDnnmMS8lFGIcRwnlhUN6o=
icc ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNpRRN65o8TcP2DnkXHdzIqAJ9CAoiz2guLSXjobx7L4meAtphb30nSx5pQqOeysU+otN9PEJH6TWr8KUXBDw6I=
exao1.magao-x.org ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMsOYTn6tlmcatxt1pDfowTtBTsmJ77OMSPl3rNl8+OBKhmpVpX+iBUMKsBDwwVIlqEAa9BfJPbSrpWEWZABv3s=
HERE
  if [[ ! $? ]]; then
    exit_with_error "Couldn't prepopulate $HOME/.ssh/known_hosts"
  fi
else
    log_info "$HOME/.ssh/known_hosts exists, not overwriting"
fi
if [[ ! -e $HOME/.ssh/config ]]; then
  cat << "HERE" | tee $HOME/.ssh/config
IdentityFile ~/.ssh/id_ed25519
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
  if [[ ! $? ]]; then
    exit_with_error "Couldn't prepopulate $HOME/.ssh/config"
  fi
else
    log_info "$HOME/.ssh/config exists, not overwriting"
fi
chmod -R u=rwX,g=,o= $HOME/.ssh/ || exit 1
