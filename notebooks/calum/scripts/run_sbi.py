import numpy as np
import sys
import argparse
import configparser
from pathlib import Path
import pickle
import torch
import time
import warnings

sys.path.append('/pbs/home/c/cmurray/cluster_likelihood/')

from modules.simulation import UniverseSimulator
from sbi.utils import BoxUniform
from sbi.inference import SNPE

def parse_args():
    parser = argparse.ArgumentParser(description='Run SBI for cluster cosmology')
    parser.add_argument('config_path', type=str, help='Path to configuration file')
    parser.add_argument('output_suffix', type=str, help='Suffix for output filename')
    parser.add_argument('--n-simulations', type=int, default=1000,
                       help='Number of simulations to run (default: 1000)')
    parser.add_argument('--output-dir', type=str,
                       default='/sps/euclid/Users/cmurray/clusters_likelihood/',
                       help='Output directory for posterior')
    return parser.parse_args()

def main():
    args = parse_args()

    print("="*70)
    print("RUNNING SBI FOR FLAGSHIP CLUSTER ANALYSIS")
    print("="*70)
    print(f"Configuration file: {args.config_path}")
    print(f"Number of simulations: {args.n_simulations}")
    print(f"Output suffix: {args.output_suffix}")
    print("="*70)

    # Load config to extract parameter names and ranges
    config = configparser.ConfigParser()
    config.read(args.config_path)

    # Define variable parameters (matching flagship_analysis.ipynb)
    variable_params = ['Omega_m', 'sigma8', 'alpha_lambda', 'beta_lambda']

    # Initialize simulator
    print("\nInitializing simulator...")
    simulator = UniverseSimulator(
        default_config_path=args.config_path,
        variable_params_names=variable_params
    )
    print("Simulator initialized successfully")

    # Define priors
    # Wide ranges for cosmology, stricter ranges for mass-richness parameters to avoid NaN issues
    prior = BoxUniform(
        low=torch.tensor([0.1, 0.5, -10.0, 0.6, 0.1]),
        high=torch.tensor([0.6, 1.0, -8.5, 0.9, 0.5])
    )

    print("\nPrior ranges:")
    prior_ranges = {
        'Omega_m': [0.1, 0.6],
        'sigma8': [0.5, 1.0],
        'alpha_lambda': [-10.0, -8.5],
        'beta_lambda': [0.6, 0.9],
        'sigma_lambda': [0.1, 0.5]
    }
    for param, bounds in prior_ranges.items():
        print(f"  {param}: [{bounds[0]}, {bounds[1]}]")

    # Run inference with SNPE (modern API)
    print(f"\nRunning SBI with {args.n_simulations} simulations...")
    print("Generating simulation samples from prior...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Sample parameters from prior
    theta_samples = prior.sample((args.n_simulations,))

    # Run simulations (this is the bottleneck - could be parallelized)
    print(f"Running simulations (this may take a while)...")
    start_time = time.time()

    # Determine summary statistics size from first simulation
    test_output = simulator.run_simulation(theta_samples[0].numpy())
    if isinstance(test_output, tuple):
        n_richness, n_redshift = test_output[0].shape
        summary_size = 2 * n_richness * n_redshift
    else:
        summary_size = len(test_output)

    x_samples = torch.zeros(args.n_simulations, summary_size, dtype=torch.float32)

    # Run simulations with progress tracking and NaN handling
    n_nan_sims = 0
    for i in range(args.n_simulations):
        if (i + 1) % 50 == 0 or i + 1 == args.n_simulations:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (args.n_simulations - i - 1)
            print(f"  Progress: {i+1}/{args.n_simulations} ({100*(i+1)/args.n_simulations:.1f}%) - "
                  f"Elapsed: {elapsed/60:.1f}m - ETA: {remaining/60:.1f}m - NaNs: {n_nan_sims}")

        theta_np = theta_samples[i].numpy()

        # Suppress warnings during simulation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            summary_stats = simulator.run_simulation(theta_np)

        # Flatten summary statistics
        if isinstance(summary_stats, tuple):
            counts, masses = summary_stats
            data_vector = np.concatenate([counts.flatten(), masses.flatten()])
        else:
            data_vector = summary_stats

        # Check for NaN or Inf values
        if np.any(np.isnan(data_vector)) or np.any(np.isinf(data_vector)):
            n_nan_sims += 1
            # Replace with zeros (will be filtered by SBI)
            data_vector = np.zeros_like(data_vector)

        x_samples[i] = torch.tensor(data_vector, dtype=torch.float32)

    total_sim_time = time.time() - start_time
    print(f"\nSimulations completed in {total_sim_time/60:.1f} minutes")
    print(f"Average time per simulation: {total_sim_time/args.n_simulations:.2f} seconds")
    print(f"Total NaN/Inf simulations: {n_nan_sims}/{args.n_simulations} ({100*n_nan_sims/args.n_simulations:.1f}%)")

    # Check if we have enough valid simulations
    valid_sims = args.n_simulations - n_nan_sims
    if valid_sims < 100:
        print(f"\nWARNING: Only {valid_sims} valid simulations!")
        print("Consider:")
        print("  1. Narrowing prior ranges further")
        print("  2. Increasing total number of simulations")
        print("  3. Checking for issues in the simulator")

    # Train neural density estimator
    print("\nTraining neural density estimator...")
    print("SBI will automatically exclude NaN/Inf simulations...")

    inference = SNPE(prior=prior)

    # Append simulations and train
    # SBI will handle NaN filtering internally
    try:
        density_estimator = inference.append_simulations(theta_samples, x_samples).train(
            training_batch_size=min(50, max(10, valid_sims // 10))  # Adaptive batch size
        )
        posterior = inference.build_posterior(density_estimator)
        print("\nSBI training completed!")
    except ValueError as e:
        print(f"\nERROR during training: {e}")
        print(f"This likely means too few valid simulations ({valid_sims}).")
        print("Try running with more simulations or narrower priors.")
        raise

    # Save posterior
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'posterior_estimator_{args.output_suffix}.pkl'

    print(f"\nSaving posterior to: {output_path}")
    with open(output_path, "wb") as handle:
        pickle.dump(posterior, handle)

    print("\nDone! Posterior saved successfully.")
    print("="*70)

if __name__ == "__main__":
    main()