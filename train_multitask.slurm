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
time python train_multitask.py --jobid $SLURM_JOB_ID \
--arch resnet18 \
--dataset cifar100 \
--add-bn-prev 0 \
--add-bn-next 1 \
--data-path ../../data/CIFAR \
--increments 10 10 10 10 10 10 10 10 10 10 \
--validation 0 \
--random-classes 0 \
--overflow 0 \
--lr 0.1 \
--weight-decay 0.0005 \
--epochs 250 \
--schedule 100 150 200 \
--compression 1.0 \
--workers 4 \
--train-batch 64 \

# python -u trainer.py  --arch=resnet56  --save-dir=checkpoints/$SLURM_JOB_ID
echo

# You're done!
echo "Ending script..."
date