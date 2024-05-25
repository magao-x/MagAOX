#!/bin/bash


pwrGUI &
dmCtrlGUI dmwoofer &
dmCtrlGUI dmtweeter &
dmCtrlGUI dmncpc &
magaox_rtimvs.sh

cameraGUI camwfs &
cameraGUI camtip &
cameraGUI camlowfs &
cameraGUI camacq &
cameraGUI camsci1 &
cameraGUI camsci2 &

loopCtrlGUI ho &
offloadCtrlGUI &
loopCtrlGUI lo &


pupilGuideGUI &

dmModeGUI wooferModes &
dmModeGUI ncpcModes &

coronAlignGUI &
sleep 6

dmnorm.sh woofer
dmnorm.sh tweeter
dmnorm.sh ncpc
