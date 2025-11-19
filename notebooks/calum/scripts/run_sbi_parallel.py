"""
Parallel SBI training script for Capish flagship analysis

This script runs simulation-based inference (SBI) using parallel simulations
to dramatically speed up the training process.

Usage:
    python run_sbi_parallel.py --n-sims 10000 --n-cores 20
    python run_sbi_parallel.py --n-sims 1000 --n-cores 10 --resume-from checkpoint.pkl
"""

import sys
import os

# IMPORTANT: Set thread limits BEFORE importing numpy/scipy/pyccl
# This prevents each multiprocessing worker from spawning too many threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import numpy as np
import pickle
from pathlib import Path
import time
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.abspath('../../../'))

import configparser
from modules.simulation_parallel import ParallelSimulator
from sbi.utils import BoxUniform
from sbi.inference import SNPE
import torch


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run parallel SBI for Capish')

    parser.add_argument('--n-sims', type=int, default=10000,
                       help='Number of simulations to run (default: 10000)')
    parser.add_argument('--n-cores', type=int, default=20,
                       help='Number of CPU cores to use (default: 20)')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                       help='Save checkpoint every N simulations (default: 1000)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints (default: ./checkpoints)')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint file')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory for outputs (default: ./outputs)')
    parser.add_argument('--output-name', type=str, default='flagship_posterior',
                       help='Name for output files (default: flagship_posterior)')
    parser.add_argument('--method', type=str, default='SNPE',
                       choices=['SNPE', 'SNLE', 'SNRE'],
                       help='SBI method (default: SNPE)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    return parser.parse_args()


def create_simulator(config_path='../../../config/capish_flagship.ini'):
    """
    Create and configure the parallel simulator.

    Parameters
    ----------
    config_path : str
        Path to configuration file

    Returns
    -------
    ParallelSimulator
        Configured simulator instance
    """
    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_path)

    # Parameters to vary (matching flagship_analysis.ipynb)
    variable_params_names = ['Omega_m', 'sigma8', 'alpha_lambda', 'beta_lambda']

    # Create parallel simulator
    simulator = ParallelSimulator(
        default_config=config,
        variable_params_names=variable_params_names
    )

    return simulator


def define_prior():
    """
    Define prior distributions for SBI.

    Returns
    -------
    torch.distributions.Distribution
        Prior distribution over parameters
    """
    # Prior ranges (widened for better parameter space coverage)
    prior_min = torch.tensor([0.15, 0.65, -10.5, 0.5])   # [Omega_m, sigma8, alpha_lambda, beta_lambda]
    prior_max = torch.tensor([0.45, 0.95, -8.0, 1.0])

    prior = BoxUniform(low=prior_min, high=prior_max)

    return prior


def sample_parameters(prior, n_sims, seed=42):
    """
    Sample parameter vectors from the prior.

    Parameters
    ----------
    prior : torch.distributions.Distribution
        Prior distribution
    n_sims : int
        Number of parameter vectors to sample
    seed : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of shape (n_sims, n_params)
    """
    torch.manual_seed(seed)
    theta_torch = prior.sample((n_sims,))
    theta_np = theta_torch.numpy()

    return theta_np


def run_simulations(simulator, theta_batch, n_cores, checkpoint_dir, args):
    """
    Run the simulation batch with checkpointing.

    Parameters
    ----------
    simulator : ParallelSimulator
        Configured simulator
    theta_batch : np.ndarray
        Parameter vectors to simulate
    n_cores : int
        Number of cores to use
    checkpoint_dir : Path
        Directory for checkpoints
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    dict
        Simulation results
    """
    checkpoint_path = checkpoint_dir / f'{args.output_name}_checkpoint.pkl'

    print(f"\n{'='*70}")
    print(f"RUNNING PARALLEL SIMULATIONS")
    print(f"{'='*70}")
    print(f"Total simulations: {len(theta_batch)}")
    print(f"CPU cores: {n_cores}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"{'='*70}\n")

    results = simulator.run_batch_parallel(
        theta_batch=theta_batch,
        n_cores=n_cores,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        desc="Simulations"
    )

    return results


def flatten_summary_stats(x):
    """
    Flatten summary statistics to 1D vectors.

    Parameters
    ----------
    x : tuple or np.ndarray
        Summary statistics (might be tuple of arrays or single array)

    Returns
    -------
    np.ndarray
        Flattened statistics, shape (n_sims, n_features)
    """
    if isinstance(x, tuple):
        # Flatten each component and concatenate
        flattened = []
        for x_i in x:
            if x_i.ndim == 1:
                flattened.append(x_i[:, np.newaxis])
            elif x_i.ndim == 2:
                flattened.append(x_i.reshape(len(x_i), -1))
            else:
                flattened.append(x_i.reshape(len(x_i), -1))
        return np.concatenate(flattened, axis=1)
    else:
        if x.ndim == 1:
            return x[:, np.newaxis]
        elif x.ndim == 2:
            return x
        else:
            return x.reshape(len(x), -1)


def train_posterior(theta, x, prior, method='SNPE'):
    """
    Train the posterior using SBI.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vectors
    x : np.ndarray
        Summary statistics
    prior : torch.distributions.Distribution
        Prior distribution
    method : str
        SBI method to use

    Returns
    -------
    posterior
        Trained posterior estimator
    """
    print(f"\n{'='*70}")
    print(f"TRAINING {method} POSTERIOR")
    print(f"{'='*70}")
    print(f"Training samples: {len(theta)}")
    print(f"Feature dimension: {x.shape[1]}")
    print(f"{'='*70}\n")

    # Convert to tensors
    theta_torch = torch.tensor(theta, dtype=torch.float32)
    x_torch = torch.tensor(x, dtype=torch.float32)

    # Train posterior using SNPE
    inference = SNPE(prior=prior)

    # Adaptive batch size based on number of samples
    training_batch_size = min(50, max(10, len(theta) // 10))

    density_estimator = inference.append_simulations(theta_torch, x_torch).train(
        training_batch_size=training_batch_size
    )
    posterior = inference.build_posterior(density_estimator)

    print(f"Training completed successfully!")

    return posterior


def save_results(results, posterior, output_dir, output_name):
    """
    Save simulation results and trained posterior.

    Parameters
    ----------
    results : dict
        Simulation results
    posterior
        Trained posterior
    output_dir : Path
        Output directory
    output_name : str
        Base name for output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save simulation results
    results_path = output_dir / f'{output_name}_simulations.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSimulation results saved: {results_path}")

    # Save posterior
    posterior_path = output_dir / f'{output_name}.pkl'
    with open(posterior_path, 'wb') as f:
        pickle.dump(posterior, f)
    print(f"Posterior saved: {posterior_path}")

    # Save summary statistics
    summary_path = output_dir / f'{output_name}_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Capish SBI Training Summary\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Total simulations: {results['n_total']}\n")
        f.write(f"Successful: {results['n_success']}\n")
        f.write(f"Failed: {results['n_failed']}\n")
        f.write(f"Success rate: {results['success_rate']:.2%}\n")
        f.write(f"Elapsed time: {results['elapsed_time']:.1f} seconds ({results['elapsed_time']/60:.1f} minutes)\n")
        f.write(f"Average time per simulation: {results['elapsed_time']/results['n_total']:.2f} seconds\n")
    print(f"Summary saved: {summary_path}")


def main():
    """Main execution function."""
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CAPISH PARALLEL SBI TRAINING")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Method: {args.method}")
    print(f"Number of simulations: {args.n_sims}")
    print(f"Number of cores: {args.n_cores}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*70}\n")

    # Create simulator
    simulator = create_simulator()

    # Define prior
    prior = define_prior()

    # Sample parameters or resume from checkpoint
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = ParallelSimulator.load_checkpoint(args.resume_from)
        results = checkpoint['results']
        n_remaining = args.n_sims - checkpoint['n_completed']
        if n_remaining > 0:
            print(f"Continuing with {n_remaining} additional simulations...")
            theta_batch = sample_parameters(prior, n_remaining, seed=args.seed + checkpoint['n_completed'])
            new_results = run_simulations(simulator, theta_batch, args.n_cores, checkpoint_dir, args)

            # Merge results
            results['theta'] = np.vstack([results['theta'], new_results['theta']])
            results['failed_theta'] = np.vstack([results['failed_theta'], new_results['failed_theta']])

            # Merge summary statistics
            if isinstance(results['x'], tuple):
                merged_x = []
                for i in range(len(results['x'])):
                    merged_x.append(np.vstack([results['x'][i], new_results['x'][i]]))
                results['x'] = tuple(merged_x)
            else:
                results['x'] = np.vstack([results['x'], new_results['x']])

            # Update statistics
            results['n_total'] += new_results['n_total']
            results['n_success'] += new_results['n_success']
            results['n_failed'] += new_results['n_failed']
            results['success_rate'] = results['n_success'] / results['n_total']
            results['elapsed_time'] += new_results['elapsed_time']
    else:
        # Sample parameters from prior
        theta_batch = sample_parameters(prior, args.n_sims, seed=args.seed)

        # Run simulations
        results = run_simulations(simulator, theta_batch, args.n_cores, checkpoint_dir, args)

    # Print results summary
    print(f"\n{'='*70}")
    print(f"SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total simulations: {results['n_total']}")
    print(f"Successful: {results['n_success']}")
    print(f"Failed (NaN/Inf): {results['n_failed']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Total time: {results['elapsed_time']:.1f} seconds ({results['elapsed_time']/60:.1f} minutes)")
    print(f"Average per simulation: {results['elapsed_time']/results['n_total']:.2f} seconds")
    print(f"{'='*70}\n")

    # Flatten summary statistics for SBI
    x_flat = flatten_summary_stats(results['x'])

    # Train posterior
    posterior = train_posterior(results['theta'], x_flat, prior, method=args.method)

    # Save results
    save_results(results, posterior, output_dir, args.output_name)

    print(f"\n{'='*70}")
    print(f"COMPLETED SUCCESSFULLY")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
