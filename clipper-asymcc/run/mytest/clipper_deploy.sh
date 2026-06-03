#!/bin/bash

USER=root
NODE_NUM=2
HOME_DIR=/root/i-NeDD/clipper-inedd/run/
CONDA_PATH=/root/anaconda3/bin/
IP_LIST=(
    172.24.128.1
    172.24.128.2
)
FILE_NAME=clipper_deploy.py
CONDA_FILE=activate
CONF_DIR=config/
CONF_FILE=simple.json
CONF_PATH=$HOME_DIR$CONF_DIR$CONF_FILE

for ip in ${IP_LIST[@]}
do
    echo ================= $ip =====================
    ssh $USER@$ip "source $CONDA_PATH$CONDA_FILE py36; cd $HOME_DIR; python3 $HOME_DIR$FILE_NAME --conf $CONF_PATH"
done