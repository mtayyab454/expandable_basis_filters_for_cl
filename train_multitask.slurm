#!/bin/bash

#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%J.out

# Give this process 1 task (per GPU, but only one GPU), then assign eight 8per task
# (so 8 cores overall).  Then enforce that slurm assigns only CPUs that are on the
# socket closest to the GPU you get.

# If you want two GPUs:
# #SBATCH --ntasks=2
# #SBATCH --gres=gpu:2

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
module load anaconda/anaconda3

# List the modules that are loaded
module listr

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m

echo

# Activate the GPU version of Pytorch
source activate my-pytorch

# OR, instead:  Activate the non-GPU version of TensorFlow
#source activate tensorflow


# Run TensorFlow
echo
#time python train_multitask.py --jobid $SLURM_JOB_ID \
#--arch alexnet_multitask \
#--add_bn 0 \
#--manualSeed 535 \
#--dataset imagenet \
#--data_path ../../../data/ImageNet \
#--increments 100 100 100 100 100 100 100 100 100 100 \
#--validation 0 \
#--random_classes 0 \
#--overflow 0 \
#--lr 0.01 \
#--weight_decay 0.0005 \
#--epochs 90 \
#--schedule 30 60 \
#--use_universal_conv 1 \
#--compression 0.50 \
#--workers 4 \
#--train_batch 64 \
#--ft_lr 0.01 \
#--ft_weight_decay 0.0005 \
#--ft_epochs 90 \
#--ft_schedule 30 60

#time python train_multitask.py --jobid $SLURM_JOB_ID \
#--arch vgg16_multitask \
#--add_bn 1 \
#--manualSeed 535 \
#--dataset tinyimagenet \
#--data_path ../../../data/tiny-imagenet-200 \
#--increments 20 20 20 20 20 20 20 20 20 20 \
#--validation 0 \
#--random_classes 0 \
#--overflow 0 \
#--lr 0.01 \
#--weight_decay 0.0005 \
#--epochs 150 \
#--schedule 70 100 120 \
#--use_universal_conv 1 \
#--compression 0.15 \
#--workers 4 \
#--train_batch 128 \
#--ft_lr 0.01 \
#--ft_weight_decay 0.0005 \
#--ft_epochs 150 \
#--ft_schedule 70 100 120

time python train_multitask.py --jobid $SLURM_JOB_ID \
--arch resnet32_multitask \
--dataset cifar100 \
--add_bn_prev 0 \
--add_bn_next 1 \
--data_path ../../../data/CIFAR \
--increments 50 50 \
--validation 0 \
--random_classes 0 \
--overflow 0 \
--lr 0.01 \
--weight_decay 0.005 \
--epochs 250 \
--schedule 100 200 \
--use_universal_conv 1 \
--compression 0.50 \
--workers 4 \
--train_batch 64 \
--ft_lr 0.01 \
--ft_weight_decay 0.005 \
--ft_epochs 250 \
--ft_schedule 100 200

#time python train_multitask.py --jobid $SLURM_JOB_ID \
#--arch resnet32_multitask \
#--dataset cifar100 \
#--add_bn_prev 0 \
#--add_bn_next 1 \
#--data_path ../../../data/CIFAR \
#--increments 50 10 10 10 10 10 \
#--validation 0 \
#--random_classes 0 \
#--overflow 0 \
#--lr 0.01 \
#--weight_decay 0.005 \
#--epochs 250 \
#--schedule 100 200 \
#--use_universal_conv 1 \
#--compression 0.15 \
#--workers 4 \
#--train_batch 64 \
#--ft_lr 0.01 \
#--ft_weight_decay 0.005 \
#--ft_epochs 250 \
#--ft_schedule 100 200

# python -u trainer.py  --arch=resnet56  --save-dir=checkpoints/$SLURM_JOB_ID
echo

# You're done!
echo "Ending script..."
date
