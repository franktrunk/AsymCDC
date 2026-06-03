#!/bin/bash

# apt
apt update
apt install -y build-essential software-properties-common

curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | apt-key add -
# sudo sh -c 'curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | apt-key add -'
add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
apt update
apt install -y docker-ce docker-ce-cli containerd.io
systemctl start docker
docker run hello-world

curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | tee /etc/apt/sources.list.d/nvidia-container-runtime.list
apt update

apt install -y nvidia-container-runtime

# install conda
wget https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate

bash Anaconda3-2022.10-Linux-x86_64.sh

source ~/.bashrc

rm -rf Anaconda3-2022.10-Linux-x86_64.sh

# create python3.6 env
conda create -n py36 python=3.6

conda activate py36

# install torch torchvision clipper_admin grpcio
pip3 install pillow==6.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install torch==1.0.1 torchvision==0.2.2 numpy==1.16.4 pandas==0.24.2
pip3 install grpcio grpcio-tools protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install ipython ipykernel