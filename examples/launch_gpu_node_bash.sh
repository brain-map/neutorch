#!/bin/bash

module load slurm
module load cuda
module load cudnn

CORES=8

srun -p gpu -N 1 --gpus 4  -C a100 --cpus-per-gpu=$CORES --pty bash -i

