#!/bin/bash
#####################################################
## MagAO-X RTC cpuset configuration 
## 
## See https://linux.die.net/man/7/cpuset
##
####################################################



#Create and mount the root cpuset
mkdir /dev/cpuset
mount -t cpuset cpuset /dev/cpuset
cd /dev/cpuset

#############################
# The system cpuset
#
# Assumes 18 cores (1 cpu, but this isn't necessary) have hyperthreading enabled, and 18 have it disabled.
# We use the hyperthread cores as the system cpuset
############################

mkdir /dev/cpuset/system
/bin/echo 0-17,36-53 > /dev/cpuset/system/cpuset.cpus
/bin/echo 0-1 > /dev/cpuset/system/cpuset.mems

# Now move all current tasks to system cpuset
# Note that this moves pid=1 (init) so all new process created should live here (right)?
#
while read i; do /bin/echo $i; done < tasks > system/tasks 

#A guess at how to setup load balancing 
echo 0 > cpuset.sched_load_balance
echo 1 > system/cpuset.sched_load_balance

###############
# Now setup cpusets for the RTC processes
# We have cores 18-35 (18 cores) to use.
# Curently unused: 35
###############

#camwfs
mkdir /dev/cpuset/camwfs
/bin/echo 18 > /dev/cpuset/camwfs/cpuset.cpus
/bin/echo 1 > /dev/cpuset/camwfs/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/camwfs/cpuset.mems

#Woofer
mkdir /dev/cpuset/dm00comb
/bin/echo 19 > /dev/cpuset/dm00comb/cpuset.cpus
/bin/echo 1 > /dev/cpuset/dm00comb/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/dm00comb/cpuset.mems

mkdir /dev/cpuset/woofer
/bin/echo 20 > /dev/cpuset/woofer/cpuset.cpus
/bin/echo 1 > /dev/cpuset/woofer/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/woofer/cpuset.mems

#Tweeter
mkdir /dev/cpuset/dm01comb
/bin/echo 21 > /dev/cpuset/dm01comb/cpuset.cpus
/bin/echo 1 > /dev/cpuset/dm01comb/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/dm01comb/cpuset.mems

mkdir /dev/cpuset/tweeter
/bin/echo 22 > /dev/cpuset/tweeter/cpuset.cpus
/bin/echo 1 > /dev/cpuset/tweeter/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/tweeter/cpuset.mems

#t2w offload process
mkdir /dev/cpuset/t2w
/bin/echo 23 > /dev/cpuset/t2w/cpuset.cpus
/bin/echo 1 > /dev/cpuset/t2w/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/t2w/cpuset.mems

#aolrun
mkdir /dev/cpuset/aol1RT
/bin/echo 24-30 > /dev/cpuset/aol1RT/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT/cpuset.mems
/bin/echo 0 > /dev/cpuset/aol1RT/cpuset.sched_load_balance

mkdir /dev/cpuset/aol1RT/aol1RT_0
/bin/echo 24 > /dev/cpuset/aol1RT/aol1RT_0/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT/aol1RT_0/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT/aol1RT_0/cpuset.mems

mkdir /dev/cpuset/aol1RT/aol1RT_1
/bin/echo 25 > /dev/cpuset/aol1RT/aol1RT_1/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT/aol1RT_1/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT/aol1RT_1/cpuset.mems

mkdir /dev/cpuset/aol1RT/aol1RT_2
/bin/echo 26 > /dev/cpuset/aol1RT/aol1RT_2/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT/aol1RT_2/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT/aol1RT_2/cpuset.mems

mkdir /dev/cpuset/aol1RT/aol1RT_3
/bin/echo 27 > /dev/cpuset/aol1RT/aol1RT_3/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT/aol1RT_3/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT/aol1RT_3/cpuset.mems

mkdir /dev/cpuset/aol1RT/aol1RT_4
/bin/echo 28 > /dev/cpuset/aol1RT/aol1RT_4/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT/aol1RT_4/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT/aol1RT_4/cpuset.mems

mkdir /dev/cpuset/aol1RT/aol1RT_5
/bin/echo 29 > /dev/cpuset/aol1RT/aol1RT_5/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT/aol1RT_5/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT/aol1RT_5/cpuset.mems

mkdir /dev/cpuset/aol1RT/aol1RT_6
/bin/echo 30 > /dev/cpuset/aol1RT/aol1RT_6/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT/aol1RT_6/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT/aol1RT_6/cpuset.mems

#aol1RT1 -- mvm extrct
mkdir /dev/cpuset/aol1RT1
/bin/echo 31 > /dev/cpuset/aol1RT1/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT1/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT1/cpuset.mems

#aol1RT2 -- gpumode2dm
mkdir /dev/cpuset/aol1RT2
/bin/echo 32,33 > /dev/cpuset/aol1RT2/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT2/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT2/cpuset.mems

/bin/echo $pid > /dev/cpuset/$cpusetname/tasks

#aol1RT3 -- aol1-ProcessModeCo
mkdir /dev/cpuset/aol1RT3
/bin/echo 34 > /dev/cpuset/aol1RT3/cpuset.cpus
/bin/echo 1 > /dev/cpuset/aol1RT3/cpuset.cpu_exclusive
/bin/echo 0-1 > /dev/cpuset/aol1RT3/cpuset.mems

