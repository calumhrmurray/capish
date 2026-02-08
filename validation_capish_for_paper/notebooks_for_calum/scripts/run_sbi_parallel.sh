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

# Capish Parallel SBI Training - SLURM Submission Script
#
# This script runs SBI training with parallel simulations using multiprocessing.
#
# Usage:
#   sbatch run_sbi_parallel.sh
#   sbatch run_sbi_parallel.sh --n-sims 5000 --n-cores 10
#
# Monitor progress:
#   tail -f sbi-logs/<job_id>.out

echo "=================================================="
echo "Capish Parallel SBI Training"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "=================================================="
echo ""

# Create log directory if it doesn't exist
mkdir -p sbi-logs
mkdir -p checkpoints
mkdir -p outputs

# Load required modules and activate conda environment
source /usr/share/Modules/init/bash
unset PYTHONPATH
module load Programming_Languages/anaconda/3.11
source activate
conda activate /sps/euclid/Users/cmurray/miniconda3/envs/capish

# Set number of threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print environment info
echo "Python version:"
python --version
echo ""
echo "Environment: $CONDA_DEFAULT_ENV"
echo ""

# Parse command-line arguments (pass through to Python script)
# Default values
N_SIMS=10000
N_CORES=$SLURM_CPUS_PER_TASK
CHECKPOINT_INTERVAL=1000
OUTPUT_NAME="flagship_posterior_${SLURM_JOB_ID}"

# Parse arguments passed to sbatch
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-sims)
            N_SIMS="$2"
            shift 2
            ;;
        --n-cores)
            N_CORES="$2"
            shift 2
            ;;
        --checkpoint-interval)
            CHECKPOINT_INTERVAL="$2"
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Build Python command
CMD="python run_sbi_parallel.py"
CMD="$CMD --n-sims $N_SIMS"
CMD="$CMD --n-cores $N_CORES"
CMD="$CMD --checkpoint-interval $CHECKPOINT_INTERVAL"
CMD="$CMD --output-name $OUTPUT_NAME"

if [ ! -z "$RESUME_FROM" ]; then
    CMD="$CMD --resume-from $RESUME_FROM"
fi

echo "Running command:"
echo "$CMD"
echo ""
echo "=================================================="
echo ""

# Run the Python script
$CMD

# Check exit status
EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"

# Calculate runtime
RUNTIME=$SECONDS
echo "Total runtime: $((RUNTIME / 60)) minutes $((RUNTIME % 60)) seconds"
echo "=================================================="

exit $EXIT_CODE
