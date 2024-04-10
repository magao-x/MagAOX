#!/usr/bin/env bash

# Import Grafana GPG key
wget -q -O gpg.key https://rpm.grafana.com/gpg.key
sudo rpm --import gpg.key

# Add Grafana repository
sudo tee /etc/yum.repos.d/grafana.repo <<EOF
[grafana]
name=grafana
baseurl=https://packages.grafana.com/oss/rpm
repo_gpgcheck=1
enabled=1
gpgcheck=1
gpgkey=https://packages.grafana.com/gpg.key
sslverify=1
sslcacert=/etc/pki/tls/certs/ca-bundle.crt
EOF

# Install Grafana
sudo yum install -y grafana

# Start Grafana service
sudo systemctl start grafana-server

# Enable Grafana service to start on boot
sudo systemctl enable grafana-server