#!/bin/bash
set -euo pipefail
if [[ -z $1 ]]; then
  role=$MAGAOX_ROLE
else
  role=$1
fi
magaox_path=/opt/MagAOX
source $magaox_path/source/MagAOX/scripts/_common.sh

filename="$magaox_path/config/proclist_$role.txt"

grep -v '^#' < $filename | { while read -r line; do
   procname=$(echo $line | awk '{print $1}')

   #if not empty, do the startup
   if [[ !  -z  ${procname// }  ]]; then
      magaox_procstart.sh $procname $role
   else
      log_warn "Empty procname in line: $line"
   fi
done }
