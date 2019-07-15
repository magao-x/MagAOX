#!/bin/bash
function log_error() {
    echo -e "$(tput setaf 1)$1$(tput sgr0)"
}
function log_success() {
    echo -e "$(tput setaf 2)$1$(tput sgr0)"
}
function log_warn() {
    echo -e "$(tput setaf 3)$1$(tput sgr0)"
}
function log_info() {
    echo -e "$(tput setaf 4)$1$(tput sgr0)"
}
# We work around the buggy devtoolset /bin/sudo wrapper in provision.sh, but
# that means we have to explicitly enable it ourselves.
# (This crap again: https://bugzilla.redhat.com/show_bug.cgi?id=1319936)
if [[ -e /opt/rh/devtoolset-7/enable ]]; then
    source /opt/rh/devtoolset-7/enable
fi
# root doesn't get /usr/local/bin on their path, so add it
# (why? https://serverfault.com/a/838552)
if [[ $PATH != "*/usr/local/bin*" ]]; then
    export PATH="/usr/local/bin:$PATH"
fi
