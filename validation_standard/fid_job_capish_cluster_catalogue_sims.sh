#!/usr/bin/bash
# SLURM options:
#SBATCH --job-name=M    # Job name
#SBATCH --output=log/%x-%j.log
#SBATCH --partition=htc               # Partition choice
#SBATCH --ntasks=9                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=3000                    # Memory in MB per default
#SBATCH --time=0-00:20:00             # 7 days by default on htc partition
#SBATCH --array=0-10
ID=$SLURM_ARRAY_TASK_ID
source /pbs/home/c/cpayerne/setup_mydesc.sh

python _run_simulations_cluster_catalogue.py $ID

