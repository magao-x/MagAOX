#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail
# set -u  # apparently makes conda angry, so just be careful about unset variables
if [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == vm || $MAGAOX_ROLE == ci ]]; then
    SUP_COMMIT_ISH=main
    orgname=magao-x
    reponame=sup
    parentdir=/opt/MagAOX/source
    clone_or_update_and_cd $orgname $reponame $parentdir
    git checkout $SUP_COMMIT_ISH
    make  # installs Python module in editable mode, builds all js (needs node/yarn)
    cd
    python -c 'import sup'  # verify sup is on PYTHONPATH

    # Install service units
    UNIT_PATH=/etc/systemd/system/
    if [[ $MAGAOX_ROLE == AOC ]]; then
        sudo cp $DIR/../systemd_units/sup.service $UNIT_PATH/sup.service
        OVERRIDE_PATH=$UNIT_PATH/sup.service.d/
        sudo mkdir -p $OVERRIDE_PATH
        echo "[Service]" | sudo tee $OVERRIDE_PATH/override.conf
        echo "Environment=\"UVICORN_HOST=0.0.0.0\"" | sudo tee -a $OVERRIDE_PATH/override.conf
        echo "Environment=\"UVICORN_PORT=4433\"" | sudo tee -a $OVERRIDE_PATH/override.conf
        echo "Environment=\"MAGAOX_ROLE=$MAGAOX_ROLE\"" | sudo tee -a $OVERRIDE_PATH/override.conf
        forwardPort="port=443:proto=tcp:toport=4433"
        if ! firewall-cmd --permanent --query-forward-port=$forwardPort; then
            sudo firewall-cmd --permanent --add-forward-port=$forwardPort
        fi
        echo "Environment=\"UVICORN_SSL_KEYFILE=/opt/sup/keys/exao1.magao-x.org.key\"" | sudo tee -a $OVERRIDE_PATH/override.conf
        echo "Environment=\"UVICORN_SSL_CERTFILE=/opt/sup/keys/exao1.magao-x.org.crt\"" | sudo tee -a $OVERRIDE_PATH/override.conf
        echo "Environment=\"UVICORN_CA_CERTS=/opt/sup/keys/exao1.magao-x.org.issuer.crt\"" | sudo tee -a $OVERRIDE_PATH/override.conf
        sudo systemctl enable sup.service || true
        sudo systemctl restart sup.service || true
    fi

    # Install localhost-only service
    sudo cp $DIR/../systemd_units/sup.service $UNIT_PATH/sup-local.service
    OVERRIDE_PATH=$UNIT_PATH/sup-local.service.d/
    sudo mkdir -p $OVERRIDE_PATH
    echo "[Service]" | sudo tee $OVERRIDE_PATH/override.conf
    echo "Environment=\"MAGAOX_ROLE=$MAGAOX_ROLE\"" | sudo tee -a $OVERRIDE_PATH/override.conf

    if [[ $MAGAOX_ROLE == vm ]]; then
        echo "Environment=\"UVICORN_HOST=0.0.0.0\"" | sudo tee -a $OVERRIDE_PATH/override.conf
        echo "User=vagrant" | sudo tee -a $OVERRIDE_PATH/override.conf
        echo "WorkingDirectory=/home/vagrant" | sudo tee -a $OVERRIDE_PATH/override.conf
    fi
    sudo systemctl enable sup-local.service || true
    sudo systemctl restart sup-local.service || true
fi
