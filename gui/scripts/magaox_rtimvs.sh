#!/bin/bash

dmdisp.sh woofer 
dmdisp.sh tweeter 
dmdisp.sh ncpc 

rtimv -c rtimv_camacq.conf &
rtimv -c rtimv_camwfs.conf &
rtimv -c rtimv_camwfs_avg.conf &
rtimv -c rtimv_camtip.conf &
rtimv -c rtimv_camlowfs.conf &
rtimv -c rtimv_camlowfs_avg.conf &
rtimv -c rtimv_camsci1.conf &
rtimv -c rtimv_camsci1_avg.conf &
rtimv -c rtimv_camsci2.conf &
rtimv -c rtimv_camsci2_avg.conf &
rtimv aol1_modevalPSDs_rawpsds &
rtimv aol1_modevalPSDs_psds &
rtimv camwfs_refsub_avg &
rtimv fdpr_phase &
rtimv fdpr_amp &

#you must include & after any rtimvs you add
