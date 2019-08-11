#!/bin/bash
set -euo pipefail

magaox_path=/opt/MagAOX
source $magaox_path/source/MagAOX/scripts/_common.sh

#!/usr/bin/bash
filename="$magaox_path/config/proclist_$1.txt"

grep -v '^#' < $filename | { while read -r line; do
   procname=$(echo $line | awk '{print $1}')

   #if not empty, do the startup
   if [[ !  -z  ${procname// }  ]]; then
      magaox_procstart.sh $procname $1
   else
      log_warn "Empty procname in line: $line"
   fi
done }
