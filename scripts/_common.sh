#!/bin/bash
function log_error() {
    echo -e "$(tput setaf 1 2>/dev/null)$1$(tput sgr0 2>/dev/null)"
}
function log_success() {
    echo -e "$(tput setaf 2 2>/dev/null)$1$(tput sgr0 2>/dev/null)"
}
function log_warn() {
    echo -e "$(tput setaf 3 2>/dev/null)$1$(tput sgr0 2>/dev/null)"
}
function log_info() {
    echo -e "$(tput setaf 4 2>/dev/null)$1$(tput sgr0 2>/dev/null)"
}
