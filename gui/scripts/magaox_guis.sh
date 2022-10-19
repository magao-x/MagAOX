#!/bin/bash

pwrGUI &
dmCtrlGUI dmwoofer &
./dmdisp.sh woofer 
dmCtrlGUI dmtweeter &
./dmdisp.sh tweeter 
dmCtrlGUI dmncpc &
./dmdisp.sh ncpc 
rtimv -c rtimv_camacq.conf &
rtimv -c rtimv_camwfs.conf &
rtimv -c rtimv_camtip.conf &
rtimv -c rtimv_camlowfs.conf &
rtimv -c rtimv_camsci1.conf &
rtimv -c rtimv_camsci2.conf &

pupilGuideGUI &
sleep 3
./dmnorm.sh woofer
./dmnorm.sh tweeter
./dmnorm.sh ncpc
