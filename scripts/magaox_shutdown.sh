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
         tmux send-keys -t $procname C-c
         tmux send-keys -t $procname "exit" Enter
      else
         echo "no running tmux session for $procname"
      fi
   else
      echo "Empty procname in line: $line"
   fi
done }
