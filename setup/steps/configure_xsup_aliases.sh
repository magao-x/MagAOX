#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -o pipefail

cat <<'HERE' > /tmp/sudoers_xsup
Defaults>xsup !env_reset
Defaults>xsup !secure_path
%magaox-dev ALL = (xsup) NOPASSWD: ALL
HERE
visudo -cf /tmp/sudoers_xsup || exit_error "visudo syntax check failed on /tmp/sudoers_xsup"
sudo mv /tmp/sudoers_xsup /etc/sudoers.d/xsup

cat <<'HERE' | sudo tee /etc/profile.d/xsupify.sh || exit 1
#!/usr/bin/env bash
alias xsupify="sudo -u xsup -i"
alias xsupdo="sudo -u xsup"
HERE