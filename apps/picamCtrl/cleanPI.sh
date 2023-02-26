#!/bin/bash

###########################################################
#
# Clean up various Picam and pleora lock and state
# files which can prevent startup of a new process
#
###########################################################



sudo rm -f /dev/shm/PrincetonInstruments\:\:Pi*
sudo rm -f /dev/shm/sem.PrincetonInstruments\:\:Pi*
sudo rm -f /dev/shm/Pits\:\:WaitForAnySharedMemory*
sudo rm -f /dev/shm/sem.*GenICam_XML
sudo rm -f /dev/shm/sem.Pits:*
sudo rm -f /var/run/pits/*
