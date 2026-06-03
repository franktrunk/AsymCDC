# AsymCDC

## Repository structure

* [model](model): Code for training and distilling the model
* [clipper-asymcc](clipper-asymcc): A prediction serving system that sits between user-facing applications and machine learning models and frameworks.It is built atop [Clipper](https://github.com/ucbrise/clipper)

## Train a Deployed Model

The deployed model is based on i-RevNet. To train a deployed model, just use the following commands. Make sure that you have installed PyTorch and CUDA in advance.

```shell
cd model/src/train

python CIFAR_main.py --nBlocks 18 18 18 --nStrides 1 2 2 --nChannels 16 64 256
```

## Distill a Decoder

The knowledge distillation is based on RepDistiller. To distill a decoder, just use the following commands. 

```shell
cd model/src/distill

python train_decoder.py --path_t ../train/checkpoint/cifar10/i-revnet-55.t7  \
                        --dataset cifar10 \
                        --distill kd \
                        --model_s iRevNet16x64 \
                        --learning_rate 0.01 \
                        -r 0.1 \
                        -a 0.9 \
                        -b 0 \
                        --trial 1 \
                        --irev \
                        --epochs 210 \
                        --lr_decay_epochs 120,150,180 \
                        --ec_k 2 \
                        --batch_size 128 
                        
```

## Run Cloud Latency Experiments

First, you should have several nodes, each of which is recommended to be equipped with at least a GPU and is enabled with CUDA>=11.4. 

### Environmet

Users can run *setup.sh* script on each node to satisfy Clipper's software requirements.

```shell
cd clipper-asymcc/run
bash scripts/setup.sh
```

### Start Cluster

Users can run clipper_deploy.sh on each node to start clipper cluster

```shell
python clipper_deploy.py
```

### Run Experiments

You can set the configuration parameters in config/*.json at first. Then run the following commands to run experiments.

```shell
python run_exp.py --conf /path/to/your/config --path /path/to/your/file
```

