#!/bin/bash
# usage: run_old.sh <outdir> [servers]  old resurrector indiserver with FIFO drivers
S=$(cd $(dirname $0); pwd); O=$1; mkdir -p $O; cd $O
IS=$S/build/oldres/INDI/INDI/indiserver
export SIMCFG=${SIMCFG:-$S/cfg.json} SIMLOG=$O
A=127.0.0.1:27624; R=127.0.0.1:27625; I=127.0.0.1:27626
for d in pdu1 observers instgraph camwfs ttmpupil holoop loloop logopt; do
  mkfifo $d.in $d.out 2>/dev/null
  ( while [ -e $O/pids ] || [ ! -e $O/stop ]; do [ -e $O/stop ] && break; $S/drv/$d < $d.in > $d.out; sleep 0.2; done ) &
done
sleep 0.5
case ${TOPO:-3} in
3)
$IS -r 1000 -p 27624 pdu1 observers instgraph camwfs@$R ttmpupil@$R holoop@$R loloop@$I logopt@$I 2> $O/isA.err & echo $! > $O/pids
$IS -r 1000 -p 27625 camwfs ttmpupil holoop pdu1@$A observers@$A loloop@$I 2> $O/isR.err & echo $! >> $O/pids
$IS -r 1000 -p 27626 loloop logopt pdu1@$A observers@$A ttmpupil@$A camwfs@$R holoop@$R 2> $O/isI.err & echo $! >> $O/pids
;;
3nochain)
$IS -r 1000 -p 27624 pdu1 observers instgraph camwfs@$R ttmpupil@$R holoop@$R loloop@$I logopt@$I 2> $O/isA.err & echo $! > $O/pids
$IS -r 1000 -p 27625 camwfs ttmpupil holoop pdu1@$A observers@$A loloop@$I 2> $O/isR.err & echo $! >> $O/pids
$IS -r 1000 -p 27626 loloop logopt pdu1@$A observers@$A ttmpupil@$R camwfs@$R holoop@$R 2> $O/isI.err & echo $! >> $O/pids
;;
2)
$IS -r 1000 -p 27624 pdu1 observers instgraph camwfs@$R ttmpupil@$R holoop@$R 2> $O/isA.err & echo $! > $O/pids
$IS -r 1000 -p 27625 camwfs ttmpupil holoop pdu1@$A observers@$A 2> $O/isR.err & echo $! >> $O/pids
;;
2oneway)
$IS -r 1000 -p 27624 pdu1 observers instgraph camwfs@$R ttmpupil@$R holoop@$R 2> $O/isA.err & echo $! > $O/pids
$IS -r 1000 -p 27625 camwfs ttmpupil holoop 2> $O/isR.err & echo $! >> $O/pids
;;
esac
