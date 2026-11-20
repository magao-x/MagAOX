#!/bin/bash
# usage: kill_old.sh <out>  stop a run_old.sh/run_fifo.sh/run_mixed.sh simulation
S=$(cd $(dirname $0); pwd); O=$1; touch $O/stop; kill $(cat $O/pids) 2>/dev/null; rm -f $O/pids; sleep 0.5; pkill -f "^[p]ython3 $S/drv/"; true
