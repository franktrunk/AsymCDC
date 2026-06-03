#!/bin/bash

cd /home/wyt/AsymCDC/clipper-asymcc/run/
source ~/anaconda3/bin/activate py36
python3 clipper_deploy.py --conf config/test.json
