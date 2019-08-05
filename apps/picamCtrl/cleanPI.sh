#!/bin/bash

###########################################################
#
# Clean up various Picam and pleora lock and state
# files which can prevent startup of a new process
#
###########################################################


set -euo pipefail

sudo rm /dev/shm/PrincetonInstruments\:\:Pi*
sudo rm /dev/shm/sem.PrincetonInstruments\:\:Pi*
sudo rm /dev/shm/Pits\:\:WaitForAnySharedMemory*
sudo rm /dev/shm/sem.*GenICam_XML
sudo rm /dev/shm/sem.Pits:*
sudo rm /var/run/pits/*
