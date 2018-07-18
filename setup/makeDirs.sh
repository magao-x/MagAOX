#!/bin/bash


##################################################
#  setup the MagAOX directory tree.
#
# This is normally configured to run on an actual MagAO-X machine.
#
# IMPORTANT: cp to the local directory and edit there if you change it
#            on your own machine.
#
# The only thing you should need to edit for use on a different machine
# is the LOGDIR definition -- probably from /data/logs to /opt/MagAOX/logs
###################################################

mkdir  /opt/MagAOX
mkdir  /opt/MagAOX/bin
mkdir  /opt/MagAOX/drivers
mkdir  /opt/MagAOX/drivers/fifos
mkdir  /opt/MagAOX/config

LOGDIR=/data/logs
#LOGDIR=/opt/MagAOX/logs

mkdir $LOGDIR
chown :magaox $LOGDIR 
chmod g+rw $LOGDIR
chmod g+s $LOGDIR

ln -s $LOGDIR /opt/MagAOX/logs

mkdir  /opt/MagAOX/sys
mkdir  /opt/MagAOX/secrets
chmod o-rwx /opt/MagAOX/secrets
chmod g-rwx /opt/MagAOX/secrets
