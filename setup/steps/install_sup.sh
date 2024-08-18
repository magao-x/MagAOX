#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -o pipefail
# set -u  # apparently makes conda angry, so just be careful about unset variables
SUP_COMMIT_ISH=main
orgname=magao-x
reponame=sup
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir
git checkout $SUP_COMMIT_ISH

if [[ ! -d /opt/conda/envs/sup ]]; then
    sudo -H /opt/conda/bin/mamba create -yn sup python pip numpy
fi
source /opt/conda/bin/activate
conda activate sup
sudo -H /opt/conda/bin/mamba env update -qf $DIR/../conda_env_sup.yml
sudo -H /opt/conda/envs/sup/bin/pip install -e /opt/MagAOX/source/purepyindi2[all]
sudo -H /opt/conda/envs/sup/bin/pip install -e /opt/MagAOX/source/magpyx
sudo -H /opt/conda/envs/sup/bin/pip install /opt/MagAOX/source/milk/src/ImageStreamIO

/opt/conda/envs/sup/bin/python -c 'import ImageStreamIOWrap' || exit 1

make  # installs Python module in editable mode, builds all js (needs node/yarn)
sudo -H /opt/conda/envs/sup/bin/pip install -e /opt/MagAOX/source/sup   # because only root can write to site-packages
cd
/opt/conda/envs/sup/bin/python -c 'import sup' || exit 1  # verify sup is on PYTHONPATH

# Install service units
UNIT_PATH=/etc/systemd/system/
if [[ $MAGAOX_ROLE == AOC ]]; then
    sudo cp $DIR/../systemd_units/sup.service $UNIT_PATH/sup.service
    OVERRIDE_PATH=$UNIT_PATH/sup.service.d/
    sudo mkdir -p $OVERRIDE_PATH
    echo "[Service]" | sudo -H tee $OVERRIDE_PATH/override.conf
    echo "Environment=\"UVICORN_HOST=0.0.0.0\"" | sudo -H tee -a $OVERRIDE_PATH/override.conf
    echo "Environment=\"UVICORN_PORT=4433\"" | sudo -H tee -a $OVERRIDE_PATH/override.conf
    echo "Environment=\"MAGAOX_ROLE=$MAGAOX_ROLE\"" | sudo -H tee -a $OVERRIDE_PATH/override.conf
    echo "Environment=\"UVICORN_SSL_KEYFILE=/home/xsup/.lego/certificates/exao1.magao-x.org.key\"" | sudo -H tee -a $OVERRIDE_PATH/override.conf
    echo "Environment=\"UVICORN_SSL_CERTFILE=/home/xsup/.lego/certificates/exao1.magao-x.org.crt\"" | sudo -H tee -a $OVERRIDE_PATH/override.conf
    echo "Environment=\"UVICORN_CA_CERTS=/home/xsup/.lego/certificates/exao1.magao-x.org.issuer.crt\"" | sudo -H tee -a $OVERRIDE_PATH/override.conf
    sudo -H firewall-cmd --add-forward-port=port=443:proto=tcp:toport=4433 --permanent
    sudo -H firewall-cmd --permanent --zone=public --add-service=https
    sudo systemctl enable sup.service || true
    sudo systemctl restart sup.service || true
fi

# Install localhost-only service
sudo cp $DIR/../systemd_units/sup.service $UNIT_PATH/sup-local.service
OVERRIDE_PATH=$UNIT_PATH/sup-local.service.d/
sudo mkdir -p $OVERRIDE_PATH
echo "[Service]" | sudo -H tee $OVERRIDE_PATH/override.conf
echo "Environment=\"MAGAOX_ROLE=$MAGAOX_ROLE\"" | sudo -H tee -a $OVERRIDE_PATH/override.conf

if [[ $VM_KIND != "none" ]]; then
    echo "Environment=\"UVICORN_HOST=0.0.0.0\"" | sudo -H tee -a $OVERRIDE_PATH/override.conf
fi
if [[ $instrument_user != xsup ]]; then
    echo "User=$instrument_user" | sudo -H tee -a $OVERRIDE_PATH/override.conf
    echo "WorkingDirectory=/home/$instrument_user" | sudo -H tee -a $OVERRIDE_PATH/override.conf
fi
sudo systemctl enable sup-local.service || true
sudo systemctl restart sup-local.service || true
