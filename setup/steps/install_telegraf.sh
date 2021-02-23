#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

cat <<EOF | sudo tee /etc/yum.repos.d/influxdb.repo
[influxdb]
name = InfluxDB Repository - RHEL \$releasever
baseurl = https://repos.influxdata.com/rhel/\$releasever/\$basearch/stable
enabled = 1
gpgcheck = 1
gpgkey = https://repos.influxdata.com/influxdb.key
EOF

sudo yum install telegraf
sudo systemctl start telegraf
if ! grep nvidia_smi /etc/telegraf/telegraf.conf; then
    sudo mv /etc/telegraf/telegraf.conf /etc/telegraf/telegraf.conf.dist
    sudo cp $DIR/../telegraf.conf /etc/telegraf/telegraf.conf.dist
fi
if [[ ! -e /etc/default/telegraf ]]; then
    log_warn "Set INFLUX_TOKEN in /etc/default/telegraf"
fi
