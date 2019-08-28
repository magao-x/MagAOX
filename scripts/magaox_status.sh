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
      if [[ ! -e $magaox_path/sys/$procname/pid ]]; then
         log_error "$procname: not started"
      else
         pid=$(cat $magaox_path/sys/$procname/pid)
         if ps -p $pid > /dev/null; then
            log_success "$procname: running (pid: $pid)"
         else
            log_error "$procname: dead (stale pid: $pid)"
         fi
      fi
      $magaox_path/bin/logdump -n 1 $procname 2>&1 | tail -n 5
   else
      log_warn "Empty procname in line: $line"
   fi
   echo
done }
