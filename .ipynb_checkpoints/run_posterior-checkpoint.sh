#!/bin/bash
#SBATCH --job-name=run_posterior
#SBATCH --output=logdir/%A_%a.out
#SBATCH --error=logdir/%A_%a.err
#SBATCH --mem=150000
#SBATCH --time=4-10:00:00             # Max time limit = 7 days
#SBATCH --ntasks=20                   # Run a single task
#SBATCH --partition=htc_highmem


# Load any necessary modules
source /usr/share/Modules/init/bash
unset PYTHONPATH
module load Programming_Languages/anaconda/3.10
source activate
conda activate sbi

python run_posteriors.py
