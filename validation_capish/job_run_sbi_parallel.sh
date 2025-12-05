#!/bin/bash
#SBATCH --job-name=capish_sbi_parallel
#SBATCH --output=sbi-logs/%j.out
#SBATCH --error=sbi-logs/%j.err
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
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

CONFIG_NARROW=narrow_prior_1_param

python sbi_run_simulations.py --config_to_simulate $CONFIG_NARROW --seed 30 --n_sims 200 --checkpoint_interval 10 --n_cores 3

N_SIMS=25000
N_CORES=$SLURM_CPUS_PER_TASK
CHECKPOINT_INTERVAL=1000
SEED=4
CONFIG=standard_prior_5_params

CMD="python sbi_run_simulations.py"
CMD="$CMD --config_to_simulate $CONFIG"
CMD="$CMD --n_cores $N_CORES"
CMD="$CMD --seed $SEED"
CMD="$CMD --n_sims $N_SIMS"
CMD="$CMD --checkpoint_interval $CHECKPOINT_INTERVAL"
echo "$CMD"

$CMD

CMD="python sbi_train_posteriors.py"
CMD="$CMD --config_to_train $CONFIG"
echo "$CMD"
$CMD