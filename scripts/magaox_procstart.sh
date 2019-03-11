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

procname=$1

#TODO: verify that line returned by grep is valid, i.e. first word matches $1, verify that execname exists,.
execname=$(grep $procname $magaox_path/config/proclist_$2.txt | awk '{print $2}')



echo "process name = $procname"
echo "executable name = $execname"

#1) check if session exists
if tmux ls | grep -q "$1"; then
   echo "Session $procname exists.  Doing nothing."
   exit 0
fi

#2) Start new session and execute command
echo "Session $procname does not exist -> creating and executing"

command="$magaox_path/bin/$execname -n $1"
echo "Executing: $command"

tmux new -s $procname -d
tmux send-keys -t $procname "$command" Enter

