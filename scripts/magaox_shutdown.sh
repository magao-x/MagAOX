#!/bin/bash
set -euo pipefail
magaox_path=/opt/MagAOX
source $magaox_path/source/MagAOX/scripts/_common.sh

#!/usr/bin/bash
filename="$magaox_path/config/proclist_$1.txt"

grep -v '^#' < $filename | { while read -r line; do
   procname=$(echo $line | awk '{print $1}')

   #if not empty, do the shutdown
   if [[ !  -z  ${procname// }  ]]; then
      if tmux ls 2>/dev/null | grep $procname > /dev/null; then
         while tmux ls 2>/dev/null | grep $procname > /dev/null; do
            tmux send-keys -t $procname C-c
            tmux send-keys -t $procname "exit" Enter
            log_info "waiting for tmux session $procname to exit..."
            sleep 0.5
         done
         log_success "Ended tmux session for $procname"
      else
         log_info "No running tmux session for $procname"
      fi
   else
      log_warn "Empty procname in line: $line"
   fi
done }
