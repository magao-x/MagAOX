#!/bin/bash


if [[ $# -eq 0 ]] ; then
    echo 'must provide DM name: woofer, tweeter, or ncpc'
    exit 1
fi

DMNAME=$1
shift #removes the name from args

case ${DMNAME} in
   woofer )
      dmindex=00
      ;;
   tweeter )
      dmindex=01
      ;;
   ncpc )
      dmindex=02
      ;;
   * )
      echo 'must provide DM name: woofer, tweeter, or ncpc'
      exit 1
esac
    
#Only use the rtimv config for the combine channel    
rtimv -c "rtimv_dm"$dmindex"disp.conf" &


for i in {0..11}
do
   if (($i < 10 ))
   then
      chnum="0$i"
   else
      chnum="$i"
   fi
   
   channel="dm"$dmindex"disp"$chnum
   
   echo $channel
   
   #remaining channels don't get the rtimv config
   rtimv --autoscale --nofpsgage $channel & 
done




