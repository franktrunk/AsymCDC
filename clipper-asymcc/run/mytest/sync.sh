#!/bin/bash

USER=root
NODE_NUM=2
HOME_DIR=/root/i-NeDD/clipper-inedd/run/
IP_LIST=(
    172.24.128.1
    172.24.128.2
)
METHOD=$1

if [ $METHOD == "config" ]
then
    CONF_DIR=config/
     
    for ip in ${IP_LIST[@]}
    do
        echo =============== $ip ===============
        for file in `ls $HOME_DIR$CONF_DIR`
        do 
            scp $HOME_DIR$CONF_DIR$file $USER@$ip:$HOME_DIR$CONF_DIR$file
            ssh $USER@$ip "sed -i \"s/\\\"local_ip\\\": \\\"${IP_LIST[0]}\\\"/\\\"local_ip\\\": \\\"$ip\\\"/\" $HOME_DIR$CONF_DIR$file"
        done
        echo ============ $ip finished =========
    done

elif [ $METHOD == "program" ]
then
    for ip in ${IP_LIST[@]}
    do
        echo =============== $ip ===============
        for file in `ls $HOME_DIR`
        do
            if [ -f $HOME_DIR$file ];
            then 
                scp $HOME_DIR$file $USER@$ip:$HOME_DIR$file 
            fi
        done
        echo ============ $ip finished =========
    done

elif [ $METHOD == "all" ]
then
    CONF_DIR=config/
     
    for ip in ${IP_LIST[@]}
    do
        echo =============== $ip ===============
        for file in `ls $HOME_DIR$CONF_DIR`
        do 
            scp $HOME_DIR$CONF_DIR$file $USER@$ip:$HOME_DIR$CONF_DIR$file
            ssh $USER@$ip "sed -i \"s/\\\"local_ip\\\": \\\"${IP_LIST[0]}\\\"/\\\"local_ip\\\": \\\"$ip\\\"/\" $HOME_DIR$CONF_DIR$file"
        done
        echo ============ $ip finished =========
    done

    for ip in ${IP_LIST[@]}
    do
        echo =============== $ip ===============
        for file in `ls $HOME_DIR`
        do
            if [ -f $HOME_DIR$file ];
            then 
                scp $HOME_DIR$file $USER@$ip:$HOME_DIR$file 
            fi
        done
        echo ============ $ip finished =========
    done

else
    echo "Error method $METHOD"
fi