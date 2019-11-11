#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

echo "export PATH=\"/usr/local/bin:\$PATH\"" | sudo tee /etc/profile.d/usr-local-bin-on-path.sh
log_success "Added /usr/local/bin to PATH via /etc/profile.d/usr-local-bin-on-path.sh"
