#!/bin/bash
# usage: run_mixed.sh <dev|stefan> <outdir>   A and I run <ver> (fork/exec drivers), R runs old resurrector (FIFO drivers)
S=$(cd $(dirname $0); pwd); V=$1; O=$2; mkdir -p $O; cd $O
NEW=$S/build/$V/INDI/INDI/indiserver; OLD=$S/build/oldres/INDI/INDI/indiserver
export PATH=$S/drv:$PATH SIMCFG=${SIMCFG:-$S/cfg.json} SIMLOG=$O
A=127.0.0.1:27624; R=127.0.0.1:27625; I=127.0.0.1:27626
for d in camwfs ttmpupil holoop; do mkfifo $d.in $d.out 2>/dev/null
  ( while [ ! -e $O/stop ]; do $S/drv/$d < $d.in > $d.out; sleep 0.2; done ) & done
sleep 0.5
$NEW -n -p 27624 pdu1 observers instgraph camwfs@$R ttmpupil@$R holoop@$R loloop@$I logopt@$I 2> $O/isA.err & echo $! > $O/pids
$OLD -r 1000 -p 27625 camwfs ttmpupil holoop pdu1@$A observers@$A loloop@$I 2> $O/isR.err & echo $! >> $O/pids
$NEW -n -p 27626 loloop logopt pdu1@$A observers@$A ttmpupil@$A camwfs@$R holoop@$R 2> $O/isI.err & echo $! >> $O/pids
