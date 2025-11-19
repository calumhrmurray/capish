# Capish Parallel SBI Pipeline

Fast parallel simulation pipeline for simulation-based inference (SBI) with Capish.

## Overview

This pipeline dramatically speeds up SBI training by running simulations in parallel using Python's multiprocessing:

- **Sequential (old)**: ~5.8 hours for 10,000 simulations
- **Parallel (new)**: ~30 minutes for 10,000 simulations with 20 cores
- **Speedup**: ~11x faster (with proper CPU allocation)

## Files

- `modules/simulation_parallel.py` - Parallel simulation wrapper class
- `run_sbi_parallel.py` - Main SBI training script with parallel simulations
- `run_sbi_parallel.sh` - SLURM submission script
- `test_parallel.py` - Quick test to verify parallel infrastructure works

## Quick Start

### 1. Test the Infrastructure

Before running a large job, verify the parallel code works:

```bash
cd notebooks/calum/scripts
python test_parallel.py
```

This runs 10 test simulations and verifies:
- Parallel infrastructure works correctly
- Speedup is achieved
- Results match between sequential and parallel runs

### 2. Run a Small Job (1,000 simulations)

Test with a smaller number of simulations first:

```bash
# Interactive test
python run_sbi_parallel.py --n-sims 1000 --n-cores 4

# Or submit to SLURM
sbatch run_sbi_parallel.sh --n-sims 1000 --n-cores 10
```

### 3. Run Full Pipeline (10,000 simulations)

```bash
sbatch run_sbi_parallel.sh
```

Default settings:
- 10,000 simulations
- 20 CPU cores
- Checkpoints every 1,000 simulations
- 2-hour time limit
- 64GB memory

## Command-Line Options

### Python Script (`run_sbi_parallel.py`)

```bash
python run_sbi_parallel.py [OPTIONS]

Options:
  --n-sims N              Number of simulations (default: 10000)
  --n-cores N             CPU cores to use (default: 20)
  --checkpoint-interval N Save checkpoint every N sims (default: 1000)
  --checkpoint-dir DIR    Checkpoint directory (default: ./checkpoints)
  --resume-from FILE      Resume from checkpoint file
  --output-dir DIR        Output directory (default: ./outputs)
  --output-name NAME      Output file name (default: flagship_posterior)
  --method {SNPE,SNLE,SNRE} SBI method (default: SNPE)
  --seed N                Random seed (default: 42)
```

### SLURM Script (`run_sbi_parallel.sh`)

```bash
sbatch run_sbi_parallel.sh [OPTIONS]

Options (passed through to Python):
  --n-sims N
  --n-cores N
  --checkpoint-interval N
  --output-name NAME
  --resume-from FILE
```

## Examples

### Example 1: Quick Test (100 simulations, 4 cores)

```bash
python run_sbi_parallel.py --n-sims 100 --n-cores 4 --output-name test_run
```

### Example 2: Medium Run (5,000 simulations, SLURM)

```bash
sbatch run_sbi_parallel.sh --n-sims 5000 --output-name flagship_5k
```

### Example 3: Resume from Checkpoint

If a job is interrupted, resume from the last checkpoint:

```bash
sbatch run_sbi_parallel.sh --resume-from checkpoints/flagship_posterior_checkpoint.pkl
```

### Example 4: Custom Prior/Parameters

Edit `run_sbi_parallel.py` to modify:
- `define_prior()` - Change prior ranges
- `create_simulator()` - Change variable parameters
- Configuration file path

## Output Files

After running, you'll find:

**In `outputs/` directory:**
- `<name>.pkl` - Trained posterior estimator (main output)
- `<name>_simulations.pkl` - All simulation results (theta, x pairs)
- `<name>_summary.txt` - Summary statistics (timing, success rate, etc.)

**In `checkpoints/` directory:**
- `<name>_checkpoint.pkl` - Latest checkpoint (auto-saved during run)

## Using the Trained Posterior

Load and use the posterior in a Jupyter notebook:

```python
import pickle
import torch
import numpy as np

# Load posterior
with open('outputs/flagship_posterior.pkl', 'rb') as f:
    posterior = pickle.load(f)

# Load flagship observations (your data)
flagship_data = np.load('flagship_cluster_catalogue_summary_statstics_DES_MoR.npy', allow_pickle=True).item()
counts = flagship_data['count_with_m200b_def']
masses = flagship_data['mean_log10m200b']

# Flatten to match training format
x_obs = np.concatenate([counts.flatten(), masses.flatten()])
x_obs_torch = torch.tensor(x_obs, dtype=torch.float32)

# Sample from posterior
samples = posterior.sample((10000,), x=x_obs_torch)

# Analyze
import matplotlib.pyplot as plt
from sbi import analysis as sbi_analysis

fig = sbi_analysis.pairplot(samples, labels=['Omega_m', 'sigma8', 'alpha_lambda', 'beta_lambda'])
plt.show()
```

## Monitoring Jobs

### Check SLURM job status

```bash
squeue -u $USER
```

### Monitor progress in real-time

```bash
tail -f sbi-logs/<job_id>.out
```

### Check resource usage

```bash
seff <job_id>
```

## Performance Tips

### 1. Optimize Number of Cores

- Too few: Slower runtime
- Too many: Overhead reduces efficiency
- Sweet spot: Usually 10-20 cores per node

Test efficiency:
```bash
python test_parallel.py  # Shows speedup and efficiency
```

### 2. Reduce NaN Rate

The current priors produce ~80% NaN simulations. To reduce waste:

**Option A: Narrow priors** (edit `define_prior()` in run_sbi_parallel.py):
```python
prior_min = torch.tensor([0.25, 0.75, -9.5, 0.7])  # Narrower ranges
prior_max = torch.tensor([0.35, 0.85, -9.0, 0.8])
```

**Option B: Use sequential rounds** (SNPE-C):
- First round: Wide prior, many NaNs
- Later rounds: Focused on high-probability regions, fewer NaNs

### 3. Checkpointing

For long runs, use frequent checkpoints:
```bash
sbatch run_sbi_parallel.sh --checkpoint-interval 500
```

Benefits:
- Resume if job times out
- Monitor progress
- Save intermediate results

## Troubleshooting

### Problem: No speedup observed

**Cause**: SLURM not allocating CPUs properly

**Fix**: Check SLURM script has:
```bash
#SBATCH --cpus-per-task=20
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

### Problem: High NaN rate (>80%)

**Cause**: Prior ranges include invalid parameter regions

**Fix**: Narrow priors or use sequential rounds of SNPE

### Problem: Out of memory

**Cause**: Too many simulations loaded in memory

**Fix**: Increase memory allocation in SLURM script:
```bash
#SBATCH --mem=128G  # Increase from 64G
```

### Problem: Job times out

**Cause**: Not enough time allocated

**Fix**: Either:
- Increase time limit: `#SBATCH --time=04:00:00`
- Use checkpointing and resume
- Reduce number of simulations

## SBI API Reference

The pipeline uses the modern `sbi` library API (v0.19+):

**Correct imports:**
```python
from sbi.utils import BoxUniform
from sbi.inference import SNPE  # Or SNLE, SNRE
```

**Training workflow:**
```python
# 1. Define prior
prior = BoxUniform(low=torch.tensor([...]), high=torch.tensor([...]))

# 2. Run simulations to get (theta, x) pairs
theta = ... # Parameter samples
x = ...     # Summary statistics

# 3. Train posterior
inference = SNPE(prior=prior)
density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)

# 4. Sample from posterior given observed data
samples = posterior.sample((n_samples,), x=x_observed)
```

**Deprecated (DO NOT USE):**
- ❌ `from sbi.inference.base import infer` - This API no longer exists
- ❌ `from sbi import utils as sbi_utils` - Use specific imports instead

## Technical Details

### Parallelization Strategy

The pipeline uses **multiprocessing** rather than MPI because:
1. Simulations are embarrassingly parallel (no inter-process communication)
2. Simpler code and better for single-node jobs
3. More efficient for shared-memory systems

### Architecture

```
Main Process
    ├── Sample parameters from prior
    ├── Create worker pool (n_cores processes)
    └── Distribute simulations to workers
         ↓
Worker Processes (each has own simulator)
    ├── Worker 1: Sim 1, 5, 9, ...
    ├── Worker 2: Sim 2, 6, 10, ...
    ├── Worker 3: Sim 3, 7, 11, ...
    └── Worker 4: Sim 4, 8, 12, ...
         ↓
Collect Results
    ├── Filter NaN/Inf
    ├── Save checkpoint
    └── Train SBI posterior
```

### Bottlenecks

Per-simulation breakdown (2.5 seconds total):
- CCL cosmology + mass function: 60-70% (1.5-1.8s)
- Observable generation: 20-30% (0.5-0.8s)
- Summary statistics: 10% (0.2-0.3s)

Potential optimizations:
- Cache CCL objects (difficult with current architecture)
- Reduce grid resolution (trade accuracy for speed)
- GPU acceleration (limited benefit for this workload)

## Comparison to Original Pipeline

| Metric | Original (`run_sbi.py`) | New (`run_sbi_parallel.py`) |
|--------|------------------------|----------------------------|
| Parallelization | None (sequential loop) | multiprocessing.Pool |
| 10K sims runtime | ~5.8 hours | ~30 minutes |
| CPU utilization | 1/20 cores = 5% | 20/20 cores = 100% |
| Checkpointing | No | Yes (every 1000 sims) |
| Resume capability | No | Yes |
| Memory usage | Low | Higher (multiple simulators) |
| Code complexity | Simple | Moderate |

## Next Steps

After training completes:

1. **Visualize posterior** in Jupyter notebook (corner plots, credible intervals)
2. **Validate coverage** - Does posterior contain true parameters?
3. **Posterior predictive checks** - Do simulated data match observations?
4. **Parameter constraints** - Extract credible intervals
5. **Comparison** - Compare to other methods (MCMC, Fisher matrix)

## Contact

For questions or issues with the parallel pipeline, check:
- SLURM logs in `sbi-logs/`
- Output summaries in `outputs/`
- Checkpoint files in `checkpoints/`
