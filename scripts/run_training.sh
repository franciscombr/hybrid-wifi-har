#!/bin/bash
#
#SBATCH --partition=gpu_min8gb  # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8gb        # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=debug_csi2har    # Job name
#SBATCH -o debug_csi2har.out       # File containing STDOUT output
#SBATCH -e debug_csi2har.err       # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python3 train_model.py --test_split 0 --val_split 0.2  --batch_size_train 24 --batch_size_val 8 --dataset_root /nas-ctm01/datasets/public/CSI-HAR/Dataset/UT_HAR_OG
