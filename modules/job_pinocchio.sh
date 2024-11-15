#!/usr/bin/bash
# SLURM options:
#SBATCH --job-name=N    # Job name
#SBATCH --output=log/%x-%j.log
#SBATCH --partition=htc               # Partition choice
#SBATCH --ntasks=3                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=7000                    # Memory in MB per default
#SBATCH --time=0-02:00:00             # 7 days by default on htc partition
#SBATCH --array=1-1           # 7 days by default on htc partition
source /pbs/home/c/cpayerne/setup_mydesc.sh

#python pinocchio_cmd.py $SLURM_ARRAY_TASK_ID

python pinocchio_generate_sky_mask_map_healpix.py
