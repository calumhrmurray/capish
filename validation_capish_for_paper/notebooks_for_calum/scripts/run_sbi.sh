#!/bin/sh

# SLURM options:
#SBATCH --job-name=flagship_sbi       # Job name
#SBATCH --output=/pbs/home/c/cmurray/cluster_likelihood/notebooks/calum/scripts/sbi-logs/%A.out
#SBATCH --error=/pbs/home/c/cmurray/cluster_likelihood/notebooks/calum/scripts/sbi-logs/%A.err
#SBATCH --partition=hpc               # Partition choice (hpc for multi-core)
#SBATCH --ntasks=1                    # Single task (Python handles multiprocessing)
#SBATCH --cpus-per-task=20            # Number of CPU cores for parallel workers
#SBATCH --mem=64000                   # Memory in MB (64GB for large simulations)
#SBATCH --time=0-12:00:00             # Max time limit (12 hours)
#SBATCH --mail-user=calum.murray@apc.in2p3.fr
#SBATCH --mail-type=END,FAIL
#SBATCH --account=euclid

# Default parameters
N_SIMULATIONS=1000
CONFIG_PATH="/pbs/home/c/cmurray/cluster_likelihood/config/capish.ini"
OUTPUT_SUFFIX="flagship"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-simulations)
            N_SIMULATIONS="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output-suffix)
            OUTPUT_SUFFIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch run_sbi.sh [--n-simulations N] [--config PATH] [--output-suffix SUFFIX]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "FLAGSHIP SBI JOB CONFIGURATION"
echo "========================================================================"
echo "Number of simulations: $N_SIMULATIONS"
echo "Config path: $CONFIG_PATH"
echo "Output suffix: $OUTPUT_SUFFIX"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "========================================================================"

# Load environment
source /usr/share/Modules/init/bash
unset PYTHONPATH
module load Programming_Languages/anaconda/3.11
source activate
conda activate /sps/euclid/Users/cmurray/miniconda3/envs/capish

# Navigate to script directory
cd /pbs/home/c/cmurray/cluster_likelihood/notebooks/calum/scripts/

# Run the Python script with command line arguments
python run_sbi.py "$CONFIG_PATH" "$OUTPUT_SUFFIX" \
    --n-simulations "$N_SIMULATIONS"

echo "========================================================================"
echo "Job completed at $(date)"
echo "========================================================================"