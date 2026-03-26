#!/bin/bash

#########################################################
# switch HOWFS MFILT input
#
# usage: howfs_mfilt_switch shmimname
#########################################################

echo 'Switching mfilt input to' $1

echo "runstop mvalC2dm-1" >> /milk/shm/tweeter-vispyr_fpsCTRL.fifo
echo "confstop mvalC2dm-1" >> /milk/shm/tweeter-vispyr_fpsCTRL.fifo
echo "runstop mfilt-1" >> /milk/shm/tweeter-vispyr_fpsCTRL.fifo
echo "confstop mfilt-1" >> /milk/shm/tweeter-vispyr_fpsCTRL.fifo

cmd='ln -sf /milk/shm/'$1'.im.shm /milk/shm/aol1_mfiltINPUT.im.shm'
$cmd

sleep 1
echo "confstart mfilt-1" >> /milk/shm/tweeter-vispyr_fpsCTRL.fifo
sleep 1
echo "runstart mfilt-1" >> /milk/shm/tweeter-vispyr_fpsCTRL.fifo
sleep 1
echo "confstart mvalC2dm-1" >> /milk/shm/tweeter-vispyr_fpsCTRL.fifo
sleep 1
echo "runstart mvalC2dm-1" >> /milk/shm/tweeter-vispyr_fpsCTRL.fifo

