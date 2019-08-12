#!/bin/bash
set -euo pipefail

############################################################
# magax_procstart.sh
#
# bash script to start a MagAO-X process in a tmux session
#
# Arguments:
#   $1 is config name
#   $2 is the system name (RTC, ICC, AOC, etc.)

magaox_path=/opt/MagAOX
source $magaox_path/source/MagAOX/scripts/_common.sh
procname=$1

#TODO: verify that line returned by grep is valid, i.e. first word matches $1, verify that execname exists,.
execname=$(grep "\b$procname\b" $magaox_path/config/proclist_$2.txt | awk '{print $2}')

log_info "process name = $procname"
log_info "executable name = $execname"

#1) check if session exists
if tmux ls | grep -q "\b$1\b"; then
   log_info "Session $procname exists.  Doing nothing."
   if [[ ! -e $magaox_path/sys/$procname/pid ]]; then
        log_error "$procname: not started"
        log_info "To inspect: tmux at -t $procname"
    else
        pid=$(cat $magaox_path/sys/$procname/pid)
        if ps -p $pid > /dev/null; then
            log_success "$procname: running (pid: $pid)"
        else
            log_error "$procname: dead (stale pid: $pid)"
            log_info "To inspect: tmux at -t $procname"
        fi
    fi
   exit 0
fi

#2) Start new session and execute command
log_info "Session $procname does not exist"

execpath="$magaox_path/bin/$execname"
if [[ ! -x $execpath ]]; then
    log_error "No executable $execpath found"
    exit 1
fi
command="$execpath -n $1"

tmux new -s $procname -d
log_info "Created tmux session '$procname'"
tmux send-keys -t $procname "$command" Enter
log_success "Executed in $procname session: $command"
