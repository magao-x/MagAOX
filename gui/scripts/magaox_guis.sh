#!/bin/bash

magaox_rtimvs.sh

pwrGUI &
dmCtrlGUI dmwoofer &
dmCtrlGUI dmtweeter &
dmCtrlGUI dmncpc &

cameraGUI camwfs &
cameraGUI camtip &
cameraGUI camflowfs &
cameraGUI camllowfs &
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

sleep 10

dmnorm.sh woofer
dmnorm.sh tweeter
dmnorm.sh ncpc
