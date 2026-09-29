#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time 100:10:00
#SBATCH --gres=gpu:1
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
 python automlrun.py
