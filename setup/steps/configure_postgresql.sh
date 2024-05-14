#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
confFile=/var/lib/pgsql/data/pg_hba.conf
dropInDir=${confFile}.d
sudo mkdir -p $dropInDir || exit_with_error "Could not make $dropInDir"
sudo chown -R postgres:postgres $dropInDir || exit_with_error "Could not set ownership of $dropInDir"
sudo chmod -R u=rwX,g=,o= $dropInDir || exit_with_error "Could not set permissions of $dropInDir"
if ! sudo grep -q "include_dir $dropInDir" $confFile; then
    log_info "Adding include directive for $dropInDir to $confFile"
    echo "include_dir $dropInDir" | sudo tee -a $confFile || exit_with_error "Could not modify $confFile"
else
    log_info "No need to add include directive to $confFile"
fi
echo "host    all             all             192.168.0.0/16            scram-sha-256" | sudo tee $dropInDir/allow_instrument_lan.conf || exit_with_error "Could not create $dropInDir/allow_instrument_lan.conf"
log_info "Added connection rule to $dropInDir/allow_instrument_lan.conf"

sudo systemctl enable postgresql.service || exit_with_error "Could not create enable postgresql service"
sudo systemctl start postgresql.service || exit_with_error "Could not start postgresql service"
# in case it was already started, make sure it reloads anyway
sudo -u postgres psql -c "SELECT pg_reload_conf();" || exit_with_error "Could not reload config as postgres user"

if [[ $MAGAOX_ROLE == AOC ]]; then
    dataArrayPath=/data/postgres
    sudo mkdir -p $dataArrayPath || exit_with_error "Could not make $dataArrayPath"
    sudo chown -R postgres:postgres $dataArrayPath || exit_with_error "Could not set ownership of $dataArrayPath"
    sudo semanage fcontext -a -t postgresql_db_t "${dataArrayPath}(/.*)?" || exit_with_error "Could not adjust SELinux context for ${dataArrayPath}"
    sudo restorecon -R ${dataArrayPath} || exit_with_error "Could not restorecon the SELinux context on ${dataArrayPath}"

    sudo -u postgres psql -c "CREATE TABLESPACE data_array LOCATION '$dataArrayPath'" || true
    sudo -u postgres psql -c "CREATE DATABASE xtelem TABLESPACE = data_array" || true
else
    sudo -u postgres psql -c "CREATE DATABASE xtelem" || true
fi
sudo -u postgres psql < $DIR/../sql/setup_users.sql || exit_with_error "Could not create database users"
