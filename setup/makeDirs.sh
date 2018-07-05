#!/bin/bash

mkdir /opt/MagAOX
mkdir /opt/MagAOX/bin
mkdir /opt/MagAOX/drivers
mkdir /opt/MagAOX/drivers/fifos
mkdir /opt/MagAOX/config

mkdir /opt/MagAOX/logs
chown :xlog /opt/MagAOX/logs
chmod g+rw /opt/MagAOX/logs
chmod g+s /opt/MagAOX/logs

mkdir /opt/MagAOX/sys
mkdir /opt/MagAOX/secrets
chmod o-rwx /opt/MagAOX/secrets
chmod g-rwx /opt/MagAOX/secrets
