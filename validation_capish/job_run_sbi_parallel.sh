#!/bin/bash
#SBATCH --job-name=capish_sbi_parallel
#SBATCH --output=sbi-logs/%j.out
#SBATCH --error=sbi-logs/%j.err
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=05:00:00

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

CONFIG=DESlike5_replicate
CONFIG_NARROW_PRIOR=_narrow_prior_1_param

# for the perfectly centered data vector
python sbi_run_simulations.py --config_to_simulate $CONFIG$CONFIG_NARROW_PRIOR --seed 30 --n_sims 300 --checkpoint_interval 10 --n_cores 3

# python sbi_run_simulations.py --config_to_simulate DESlike3_MoR_log10Mwl_stacked_scatter_narrow_prior_1_param --seed 30 --n_sims 300 --checkpoint_interval 10 --n_cores 3

# for the 5 params mcmc
N_SIMS=50000
N_CORES=$SLURM_CPUS_PER_TASK
CHECKPOINT_INTERVAL=1000
SEED=49
CONFIG_LARGE_PRIOR=_standard_prior_6_params

CMD="python sbi_run_simulations.py"
CMD="$CMD --config_to_simulate $CONFIG$CONFIG_LARGE_PRIOR"
CMD="$CMD --n_cores $N_CORES"
CMD="$CMD --seed $SEED"
CMD="$CMD --n_sims $N_SIMS"
CMD="$CMD --checkpoint_interval $CHECKPOINT_INTERVAL"
echo "$CMD"

$CMD

CMD="python sbi_train_posteriors.py"
CMD="$CMD --config_to_train $CONFIG$CONFIG_LARGE_PRIOR"
echo "$CMD"
$CMD

# python sbi_train_posteriors.py --config_to_train DESlike4_replicate_standard_prior_6_params