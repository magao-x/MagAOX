
pidlist=$( pgrep DMcomb-000000.r | xargs echo | sed 's/ /,/g' )
echo "DMcomb 0            : $pidlist"
/bin/echo $pidlist > /dev/cpuset/dm00comb/tasks
echo ""

pidlist=$( pgrep DMcomb-000001.r | xargs echo | sed 's/ /,/g' )
echo "DMcomb 1            : $pidlist"
/bin/echo $pidlist > /dev/cpuset/dm01comb/tasks
echo ""

pidlist=$( pgrep aol1run | xargs echo | sed 's/ /,/g' )
echo "aol1 (master)       : $pidlist"
aolPID="$pidlist"
echo ""

pidlist=$( ls /proc/$aolPID/task/ )
echo "aol1 (full) :"
echo "$pidlist"

let i=0
for pid in $pidlist
do
   /bin/echo $pid > /dev/cpuset/aol1RT/aol1RT_${i}/tasks
   let i=i+1
done
echo ""



pidlist=$( pgrep aol1mexwfs | xargs echo | sed 's/ /,/g' )
echo "cudaMVMextract aol1 : $pidlist"
aolPID="$pidlist"
/bin/echo $pidlist > /dev/cpuset/aol1RT1/tasks
echo ""


pidlist=$( pgrep aol1dmfw | xargs echo | sed 's/ /,/g' )
echo "aol1 GPUmodes2dm    : $pidlist"
aolPID="$pidlist"
echo ""
pidlist=$( ls /proc/$aolPID/task/ )
echo "aol1 GPUmodes2dm (full) :"
echo "$pidlist"
for pid in $pidlist
do
   /bin/echo $pid > /dev/cpuset/aol1RT2/tasks
done
echo ""

pidlist=$( pgrep aol1meol | xargs echo | sed 's/ /,/g' )
echo "aol1meol         : $pidlist"
/bin/echo $pidlist > /dev/cpuset/aol1RT3/tasks
echo ""

