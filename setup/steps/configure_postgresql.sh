#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
sed -i "s/^#*listen_addresses.*/listen_addresses = '*'/" /var/lib/pgsql/data/postgresql.conf
log_info "Bound to all listen addresses in /var/lib/pgsql/data/postgresql.conf"

if [[ ! -e /var/lib/pgsql/data/pg_hba.conf.dist ]]; then
    sudo cp /var/lib/pgsql/data/pg_hba.conf /var/lib/pgsql/data/pg_hba.conf.dist
sudo tee /var/lib/pgsql/data/pg_hba.conf <<EOF
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# "local" is for Unix domain socket connections only
local   all             all                                     peer
# IPv4 local connections:
# host    all             all             127.0.0.1/32            scram-sha-256
# IPv6 local connections:
# host    all             all             ::1/128                 scram-sha-256
# Allow replication connections from localhost, by a user with the
# replication privilege.
local   replication     all                                     peer
host    replication     all             127.0.0.1/32            scram-sha-256
host    replication     all             ::1/128                 scram-sha-256
host    all             all             192.168.0.0/16          scram-sha-256
host    all             all             ::1/128                 scram-sha-256
host    all             all             127.0.0.1/32            scram-sha-256
EOF
fi
sed -i "s/^#*listen_addresses.*/listen_addresses = '*'/" /var/lib/pgsql/data/postgresql.conf

sudo systemctl enable postgresql.service || exit_with_error "Could not create enable postgresql service"
sudo systemctl restart postgresql.service || exit_with_error "Could not start postgresql service"

if [[ $MAGAOX_ROLE == AOC ]]; then
    dataArrayPath=/data/postgres
    sudo mkdir -p $dataArrayPath || exit_with_error "Could not make $dataArrayPath"
    sudo chown -R postgres:postgres $dataArrayPath || exit_with_error "Could not set ownership of $dataArrayPath"

    sudo tee /etc/systemd/system/var-lib-pgsql-extdata.mount <<EOF
[Unit]
Description=Bind Mount for /var/lib/pgsql/extdata
Before=postgresql.service

[Mount]
What=/data/postgres/tablespace
Where=/var/lib/pgsql/extdata
Type=none
Options=bind

[Install]
WantedBy=multi-user.target

[Service]
ExecStart=/sbin/restorecon -Rv /var/lib/pgsql/extdata
EOF

    sudo tee /etc/systemd/system/var-lib-pgsql-extdata.automount <<EOF
[Unit]
Description=Automount for /var/lib/pgsql/extdata

[Automount]
Where=/var/lib/pgsql/extdata
TimeoutIdleSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo semanage fcontext -a -t postgresql_db_t "${dataArrayPath}(/.*)?" || exit_with_error "Could not adjust SELinux context for ${dataArrayPath}"
    sudo restorecon -R ${dataArrayPath} || exit_with_error "Could not restorecon the SELinux context on ${dataArrayPath}"

    sudo -u postgres psql -c "CREATE TABLESPACE data_array LOCATION '$dataArrayPath'" || true
    sudo -u postgres psql -c "CREATE DATABASE xtelem WITH OWNER = xtelem TABLESPACE = data_array" || true
else
    sudo -u postgres psql -c "CREATE DATABASE xtelem" || true
fi
sudo -u postgres psql < $DIR/../sql/setup_users.sql || exit_with_error "Could not create database users"
