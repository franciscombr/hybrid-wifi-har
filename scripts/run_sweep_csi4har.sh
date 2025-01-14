#!/bin/bash
#
#SBATCH --partition=gpu_min8gb  # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8gb        # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=sweep_csi4har    # Job name
#SBATCH -o logs/sweep_csi4har.out       # File containing STDOUT output
#SBATCH -e logs/sweep_csi4har.err       # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
python3 train_csi4har.py
