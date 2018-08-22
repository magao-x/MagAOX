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

mkdir  -pv /opt/MagAOX
mkdir  -pv /opt/MagAOX/bin
mkdir  -pv /opt/MagAOX/drivers
mkdir  -pv /opt/MagAOX/drivers/fifos
mkdir  -pv /opt/MagAOX/config

mkdir -pv "$LOGDIR"
if [[ ! $(getent group magaox) ]]; then
  groupadd magaox
  echo "Added group magaox"
else
  echo "Group magaox exists"
fi
chown :magaox "$LOGDIR"
chmod g+rw -v "$LOGDIR"
chmod g+s -v "$LOGDIR"

if [ "$LOGDIR" != "/opt/MagAOX/logs" ] ; then
  echo "Creating logs symlink . . ."
  ln -s $LOGDIR /opt/MagAOX/logs
fi

mkdir  -pv /opt/MagAOX/sys
mkdir  -pv /opt/MagAOX/secrets
chmod o-rwx -v /opt/MagAOX/secrets
chmod g-rwx -v /opt/MagAOX/secrets
