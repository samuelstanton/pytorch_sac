# MBPO implementation in PyTorch
Built on a modified version of `pytorch_sac` (https://github.com/denisyarats/pytorch_sac)

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment and activate it:
```
conda env create -f conda_env.yml
source activate pytorch_mbpo
```

## Instructions
To train an MBPO agent on the `Hopper-v2` task in OpenAI Gym, run:
```
python train_mbpo.py task=gym_hopper fwd_model=fc_ensemble
```
This will produce `exp` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir exp
```
