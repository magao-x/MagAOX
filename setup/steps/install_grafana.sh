#!/usr/bin/env bash

# Install Grafana
# Check if Grafana is already installed
if ! command -v grafana-server &> /dev/null; then
    # Import Grafana GPG key
    wget -q -O /tmp/gpg.key https://rpm.grafana.com/gpg.key
    sudo rpm --import /tmp/gpg.key

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
    sudo dnf install -y grafana
    sudo cp /etc/grafana/grafana.ini /etc/grafana/grafana.ini.dist
fi

sudo tee /etc/grafana/grafana.ini <<EOF
[paths]
provisioning = /opt/MagAOX/source/MagAOX/setup/grafana
[security]
admin_user = vizzy
admin_password = extremeAO!
[users]
allow_sign_up = false
[auth.anonymous]
enabled = true
[date_formats]
default_timezone = UTC
EOF

# Enable Grafana service to start on boot
sudo systemctl enable grafana-server

# Start Grafana service
sudo systemctl restart grafana-server