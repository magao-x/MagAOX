#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

echo "export PATH=\"\$PATH:/usr/local/bin\"" | sudo tee /etc/profile.d/usr-local-bin-on-path.sh
log_success "Added /usr/local/bin to PATH via /etc/profile.d/usr-local-bin-on-path.sh"

sudo chmod +x /etc/rc.d/rc.local
log_success "Made /etc/rc.d/rc.local executable to silence console messages from systemd-rc-local-generator"