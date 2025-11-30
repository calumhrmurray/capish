#!/bin/bash
#SBATCH --job-name=capish_sbi_parallel
#SBATCH --output=sbi-logs/%j.out
#SBATCH --error=sbi-logs/%j.err
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Load required modules and activate conda environment
source /usr/share/Modules/init/bash
unset PYTHONPATH
module load Programming_Languages/anaconda/3.11
source activate
conda activate /sps/lsst/users/cpayerne/envs/sbi_env

# Set number of threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

N_SIMS=10000
N_CORES=$SLURM_CPUS_PER_TASK
CHECKPOINT_INTERVAL=1000
SEED=1

CMD="python run_sbi_parallel_from_config_posterior_training.py"
CMD="$CMD --config_to_train standard_prior_6_params"
CMD="$CMD --n_cores $N_CORES"
CMD="$CMD --seed $SEED"
CMD="$CMD --n_sims $N_SIMS"
CMD="$CMD --checkpoint_interval $CHECKPOINT_INTERVAL"
echo "$CMD"
$CMD
