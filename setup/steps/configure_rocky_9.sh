#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

echo "export PATH=\"\$PATH:/usr/local/bin\"" | sudo tee /etc/profile.d/usr-local-bin-on-path.sh
log_success "Added /usr/local/bin to PATH via /etc/profile.d/usr-local-bin-on-path.sh"

sudo chmod +x /etc/rc.d/rc.local
log_success "Made /etc/rc.d/rc.local executable to silence console messages from systemd-rc-local-generator"

sudo mkdir -p /usr/local/lib64
echo "/usr/local/lib64" | sudo tee /etc/ld.so.conf.d/lib64.conf
log_success "Made /usr/local/lib64 and corresponding /etc/ld.so.conf.d/lib64.conf"
