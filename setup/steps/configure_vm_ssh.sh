#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
#
# Pre-populate known hosts and hostname aliases for SSH tunneling from the VM
#
if [[ $VM_KIND == "vagrant" ]]; then
  username=vagrant
else
  username=$USER
fi
if [[ $MAGAOX_ROLE == vm && $VM_WINDOWS_HOST == 0 ]]; then
  mkdir -p "$VM_SHARED_FOLDER/ssh"
  if [[ ! -e "$VM_SHARED_FOLDER/ssh/config" ]]; then
    cat << "HERE" | sudo tee "$VM_SHARED_FOLDER/ssh/config"
IdentityFile $VM_SHARED_FOLDER/ssh/magaox_ssh_key
Host aoc
  HostName exao1.magao-x.org
Host rtc
  HostName rtc
  ProxyJump aoc
Host icc
  HostName icc
  ProxyJump aoc
Host tic
  HostName exao0.as.arizona.edu
Host *
  User YOURMAGAOXUSERNAME
HERE
  fi
  mkdir -p /home/$username/.ssh
  if [[ ! -e /home/$username/.ssh/known_hosts ]]; then
      cat <<'HERE' | sudo tee /home/$username/.ssh/known_hosts
rtc ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBFmgoTzcAVYXDZjPFNLfpPz/T/0DQvrXSe9XOly9SD7NcjwN/fRTk+DhrWzdPN5aBsDnnmMS8lFGIcRwnlhUN6o=
icc ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNpRRN65o8TcP2DnkXHdzIqAJ9CAoiz2guLSXjobx7L4meAtphb30nSx5pQqOeysU+otN9PEJH6TWr8KUXBDw6I=
exao1.magao-x.org ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMsOYTn6tlmcatxt1pDfowTtBTsmJ77OMSPl3rNl8+OBKhmpVpX+iBUMKsBDwwVIlqEAa9BfJPbSrpWEWZABv3s=
HERE
  else
      log_info "/home/$username/.ssh/known_hosts exists, not overwriting"
  fi
  if [[ ! -e /home/$username/.ssh/config ]]; then
    cat << "HERE" | sudo tee /home/$username/.ssh/config
Include $VM_SHARED_FOLDER/ssh/config
HERE
  else
      log_info "/home/$username/.ssh/config exists, not overwriting"
  fi
  sudo chown -R $username:$username /home/$username/.ssh/
  sudo chmod -R u=rwX,g=,o= /home/$username/.ssh/
fi
