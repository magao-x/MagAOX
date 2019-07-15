#!/bin/bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/_common.sh

$DIR/add_a_user.sh $1
gpasswd -a $1 magaox-dev
gpasswd -a $1 wheel
log_success "Added $1 to groups magaox-dev and wheel"
