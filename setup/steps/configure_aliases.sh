#!/usr/bin/env bash
if [[ "$EUID" != 0 ]]; then
    sudo -H bash $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -o pipefail

echo "alias teldump='logdump --dir=/opt/MagAOX/telem --ext=.bintel'" | sudo tee /etc/profile.d/teldump.sh

cat <<'HERE' > /tmp/sudoers_xsup
# keep MAGAOX_ROLE set for any sudo'd command
Defaults env_keep += "MAGAOX_ROLE"
# keep entire environment when becoming xsup
Defaults>xsup !env_reset
Defaults>xsup !secure_path
# disable password authentication to become xsup
%magaox ALL = (xsup) NOPASSWD: ALL
%magaox ALL = (root) NOPASSWD: /opt/MagAOX/bin/write_magaox_pidfile
HERE
visudo -cf /tmp/sudoers_xsup || exit_with_error "visudo syntax check failed on /tmp/sudoers_xsup"
sudo mv /tmp/sudoers_xsup /etc/sudoers.d/xsup

cat <<'HERE' | sudo tee /etc/profile.d/xsupify.sh || exit 1
#!/usr/bin/env bash
alias xsupify="/usr/bin/sudo -u xsup -i"
alias xsupdo="/usr/bin/sudo -u xsup"
HERE