#!/bin/bash


if [[ $# -eq 0 ]] ; then
    echo 'must provide DM name: woofer, tweeter, or ncpc'
    exit 1
fi

DMNAME=$1
shift #removes the name from args

#Change dy0 to 0+ to move dmdisps to top of screen
#dy0 = 1275 + puts them at the bottom of the screen

#for bottom row att 2165 to all y
case ${DMNAME} in
   woofer )
      dmctrl=dmwoofer
      dmindex=00
      dx0=0
      dy0=1275
      ;;
   tweeter )
      dmctrl=dmtweeter
      dmindex=01
      dx0=0
      dy0=1275+298
      ;;
   ncpc )
      dmctrl=dmncpc
      dmindex=02
      dx0=0
      dy0=1275+596
      ;;
   * )
      echo 'must provide DM name: woofer, tweeter, or ncpc'
      exit 1
esac
    
#for 4 monitors:    
x0=3840
#for 3 monitors:
#x0=7680


y0=26
dmctrlWidth=475
dmdispSize=259


while getopts ":x:y:w:s:" opt; do
   case ${opt} in
     x )
       x0=$OPTARG
       ;;
     y )
       y0=$OPTARG
       ;;
     w )
       dmctrlWidth=$OPTARG
       ;;
     s )
       dmdispSize=$OPTARG
       ;;
     \? )
       echo "Invalid Option: -$OPTARG" 1>&2
       exit 1
       ;;
     : )
       echo "Invalid Option: -$OPTARG requires an argument" 1>&2
       exit 1
       ;;
   esac
done

x0=$((x0+dx0))
y0=$((y0+dy0))

echo "Normalizing $DMNAME with: "
echo "   Control:   $dmctrl"
echo "   Index:     $dmindex"
echo "   x0:        $x0"
echo "   y0:        $y0"
echo "   Control w: $dmctrlWidth"
echo "   Disp size: $dmdispSize"

wmctrl -F -r $dmctrl   -e 0,$x0,$y0,$dmctrlWidth,$dmdispSize

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
   
   wmctrl -F -r $channel -e 0,$(( x0+$dmctrlWidth+i*$dmdispSize)),$y0,$dmdispSize,$dmdispSize
done

wmctrl -F -r "dm"$dmindex"disp" -e 0,$(( x0+$dmctrlWidth+12*$dmdispSize)),$y0,$dmdispSize,$dmdispSize



