#!/bin/sh

# SLURM options:
#SBATCH --job-name=run_des_y1_analysis    # Job name
#SBATCH --output=/sps/euclid/Users/cmurray/sel_function_logs/logdir/%A_%a.out
#SBATCH --error=/sps/euclid/Users/cmurray/sel_function_logs/logdir/%A_%a.err
#SBATCH --partition=htc_interactive               # Partition choice (htc by default)
#SBATCH --ntasks=32                    # Run a single task
#SBATCH --mem=10000                   # Memory in MB per default
#SBATCH --time=0-4:00:00             # Max time limit = 7 days
#SBATCH --mail-user=calum.murray@apc.in2p3.fr   # Where to send mail
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --account=euclid

# Commands to be submitted:
source /usr/share/Modules/init/bash
unset PYTHONPATH
module load Programming_Languages/anaconda/3.11
source activate
conda activate sbi

# Run the Python script with the provided settings file name
python DESy1_infer_script_halo_model.py
