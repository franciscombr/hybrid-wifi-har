#!/bin/bash
#
#SBATCH --partition=gpu_min8gb  # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8gb        # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=debug_resnet    # Job name
#SBATCH -o debug_resnet.out       # File containing STDOUT output
#SBATCH -e debug_resnet.err       # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python3 ../src/resnet/train_utils_resnet.py --normalize --learning_rate 0.0001 --num_epochs 50 --test_split 0.2 --val_split 0.2  --batch_size_train 24 --batch_size_val 8 --dataset_root /nas-ctm01/datasets/public/CSI-HAR/Dataset/UT_HAR_OG 
