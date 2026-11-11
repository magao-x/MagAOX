#!/bin/bash
# usage: kill.sh <out>  stop a run.sh/run_exit.sh simulation
S=$(cd $(dirname $0); pwd); O=$1; kill $(cat $O/pids) 2>/dev/null; pkill -f "^$S/build/[a-z]*/INDI/INDI/indiserver "; sleep 0.5; pkill -f "^[p]ython3 $S/drv/"; true
