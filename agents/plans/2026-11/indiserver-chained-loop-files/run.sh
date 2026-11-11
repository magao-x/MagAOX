#!/bin/bash
# usage: run.sh <dev|stefan> <outdir>   starts A(27624) R(27625) I(27626); kill with kill.sh
S=$(cd $(dirname $0); pwd); V=$1; O=$2; mkdir -p $O
IS=$S/build/$V/INDI/INDI/indiserver
FL="-n"; [ "$V" = oldres ] && FL="-r 1000"
export PATH=$S/drv:$PATH SIMCFG=${SIMCFG:-$S/cfg.json} SIMLOG=$O
A=127.0.0.1:27624; R=127.0.0.1:27625; I=127.0.0.1:27626
cd $O
$IS $FL -p 27624 pdu1 observers instgraph camwfs@$R ttmpupil@$R holoop@$R loloop@$I logopt@$I 2> $O/isA.err & echo $! > $O/pids
$IS $FL -p 27625 camwfs ttmpupil holoop pdu1@$A observers@$A loloop@$I 2> $O/isR.err & echo $! >> $O/pids
$IS $FL -p 27626 loloop logopt pdu1@$A observers@$A ttmpupil@$A camwfs@$R holoop@$R 2> $O/isI.err & echo $! >> $O/pids
