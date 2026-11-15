#!/bin/bash
# restart server R (2nd pid) in place
S=$(cd $(dirname $0); pwd); V=$1; O=$2
IS=$S/build/$V/INDI/INDI/indiserver
FL="-n"; [ "$V" = oldres ] && FL="-r 1000"
export PATH=$S/drv:$PATH SIMCFG=${SIMCFG:-$S/cfg.json} SIMLOG=$O
A=127.0.0.1:27624; R=127.0.0.1:27625; I=127.0.0.1:27626
P=$(sed -n 2p $O/pids); kill $P; sleep 0.3; pkill -f "^[p]ython3 $S/drv/(camwfs|ttmpupil|holoop)$"; sleep ${3:-1}
cd $O; $IS $FL -p 27625 camwfs ttmpupil holoop pdu1@$A observers@$A loloop@$I 2>> $O/isR.err &
sed -i "2s/.*/$!/" $O/pids
