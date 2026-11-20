#!/bin/bash
# usage: run_fifo.sh <dev|stefan> <outdir>  Stefan-branch FIFO driver mode (-F), TOPO=3
S=$(cd $(dirname $0); pwd); V=$1; O=$2; mkdir -p $O/dA $O/dR $O/dI; cd $O
IS=$S/build/$V/INDI/INDI/indiserver
export SIMCFG=${SIMCFG:-$S/cfg.json} SIMLOG=$O
A=127.0.0.1:27624; R=127.0.0.1:27625; I=127.0.0.1:27626
att() { for d in "${@:2}"; do mkfifo $1/$d.in $1/$d.out 2>/dev/null
  ( while [ ! -e $O/stop ]; do $S/drv/$d > $1/$d.out < $1/$d.in; sleep 0.2; done ) & done; }
att dA pdu1 observers instgraph; att dR camwfs ttmpupil holoop; att dI loloop logopt
$IS -F dA -p 27624 pdu1 observers instgraph camwfs@$R ttmpupil@$R holoop@$R loloop@$I logopt@$I 2> $O/isA.err & echo $! > $O/pids
$IS -F dR -p 27625 camwfs ttmpupil holoop pdu1@$A observers@$A loloop@$I 2> $O/isR.err & echo $! >> $O/pids
$IS -F dI -p 27626 loloop logopt pdu1@$A observers@$A ttmpupil@$A camwfs@$R holoop@$R 2> $O/isI.err & echo $! >> $O/pids
