#!/bin/bash
set -euo pipefail

magaox_path=/opt/MagAOX

#!/usr/bin/bash
filename="$magaox_path/config/proclist_$1.txt"

while read -r line; do
   nocomment=${line%%#*}
   procname=$(echo $nocomment | awk '{print $1}')
   
   #if not empty, do the shutdown
   if [[ !  -z  ${procname// }  ]]; then
      tmux send-keys -t $procname C-c
      tmux send-keys -t $procname "exit" Enter
   fi
done < "$filename"
