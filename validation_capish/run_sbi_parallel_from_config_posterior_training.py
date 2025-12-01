import sys, os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
import pickle
import argparse
import time
from datetime import datetime
from pathlib import Path
import configparser

from sbi.utils import BoxUniform
from sbi.inference import SNPE

import config_posterior_training

sys.path.append('../')
from modules.simulation_parallel import ParallelSimulator

def parse_args():

    parser = argparse.ArgumentParser(description='Run parallel SBI for Capish')
    parser.add_argument('--config_to_train', type=str, default='', help='which to train ?')
    parser.add_argument('--seed', type=int, default=20, help='seed')
    parser.add_argument('--n_sims', type=int, default=20, help='number of simulations')
    parser.add_argument('--n_cores', type=int, default=20, help='Number of CPU cores to use (default: 20)')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='checkpoint_interval')

    return parser.parse_args()

def create_simulator(var_names, config_path, default_config):
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return ParallelSimulator(default_config=cfg,config_path=config_path,
                                    variable_params_names=var_names)

def define_prior(prior_min, prior_max):
    return BoxUniform(
        low=torch.tensor(prior_min),
        high=torch.tensor(prior_max))

def sample_parameters(prior, n, seed):
    torch.manual_seed(seed)
    return prior.sample((n,)).numpy()

def flatten_summary_stats(x):
    if isinstance(x, tuple):
        flattened = [xi.reshape(len(xi), -1) for xi in x]
        return np.concatenate(flattened, axis=1)
    return x.reshape(len(x), -1)

def run_simulations(sim, theta, n_cores, ckpt_dir, output_name, checkpoint_interval):
    ckpt_path = ckpt_dir / f"{output_name}_checkpoint.pkl"
    print(f"\nRunning {len(theta)} parallel simulations on {n_cores} cores.")

    return sim.run_batch_parallel(
        theta_batch=theta,
        n_cores=n_cores,
        checkpoint_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        desc="Simulations")

def train_posterior(theta, x, prior, method="SNPE"):
    print(f"\nTraining {method} posterior on {len(theta)} samples…")

    theta = torch.tensor(theta, dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)

    inference = SNPE(prior=prior)

    batch_size = min(50, max(10, len(theta) // 10))
    density_est = inference.append_simulations(theta, x).train(
                        training_batch_size=batch_size)

    return inference.build_posterior(density_est)

def save_results(results, posterior, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{name}simulations.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(out_dir / f"{name}posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)

    with open(out_dir / f"{name}summary.txt", "w") as f:
        f.write("Capish SBI Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total simulations: {results['n_total']}\n")
        f.write(f"Successful: {results['n_success']}\n")
        f.write(f"Failed: {results['n_failed']}\n")
        f.write(f"Success rate: {results['success_rate']:.2%}\n")
        f.write(f"Elapsed time: {results['elapsed_time']:.1f}s\n")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    args = parse_args()
    name = args.config_to_train
    n_sims = args.n_sims
    ncores = int(args.n_cores)
    seed = int(args.seed)
    checkpoint_interval = int(args.checkpoint_interval)

    
    cfg = config_posterior_training.config_dict[name]
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_dir = Path(cfg['checkpoint_dir'])
    output_dir = Path(cfg['output_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_name = cfg["output_name"]

    print("\n" + "=" * 70)
    print("CAPISH PARALLEL SBI TRAINING")
    print("=" * 70)
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Method: {cfg['method']}")
    print(f"Simulations: {n_sims}")
    print(f"Cores: {ncores}")
    print("=" * 70)

    simulator = create_simulator(
        cfg['variable_params_names'],
        cfg["config.ini_path"],
        cfg["config.ini"])
    
    prior = define_prior(cfg['prior_min'], cfg['prior_max'])

    # Resume or run fresh
    if cfg['resume_from']:
        ckpt = ParallelSimulator.load_checkpoint(cfg['resume_from'])
        results = ckpt['results']

        remaining = n_sims - ckpt['n_completed']
        if remaining > 0:
            theta = sample_parameters(prior, remaining, seed + ckpt['n_completed'])
            new = run_simulations(simulator, theta, ncores, checkpoint_dir, output_name, checkpoint_interval)

            results['theta'] = np.vstack([results['theta'], new['theta']])
            results['failed_theta'] = np.vstack([results['failed_theta'], new['failed_theta']])

            if isinstance(results['x'], tuple):
                results['x'] = tuple(
                    np.vstack([results['x'][i], new['x'][i]])
                    for i in range(len(results['x'])))
            else:
                results['x'] = np.vstack([results['x'], new['x']])

            for k in ['n_total', 'n_success', 'n_failed', 'elapsed_time']:
                results[k] += new[k]

            results['success_rate'] = results['n_success'] / results['n_total']

    else:
        theta = sample_parameters(prior, n_sims, seed)
        results = run_simulations(simulator, theta, ncores, checkpoint_dir, output_name, checkpoint_interval)

    print("Simulation Summary")
    print("=" * 70)
    print(f"Total: {results['n_total']}")
    print(f"Success: {results['n_success']}")
    print(f"Failed: {results['n_failed']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Time: {results['elapsed_time']:.1f}s")
    print("=" * 70)

    x_count_log10m = results['x']
    x_count, x_log10mass = x_count_log10m
    x_count_Nmass = (x_count, x_count * 10 ** x_log10mass)

    x_flat_count_log10m = flatten_summary_stats(x_count_log10m)
    x_flat_count = flatten_summary_stats(x_count)
    x_flat_log10mass = flatten_summary_stats(x_log10mass)
    x_flat_count_Nmass = flatten_summary_stats(x_count_Nmass)
    
    posterior_count_log10m = train_posterior(results['theta'], x_flat_count_log10m, prior, cfg['method'])
    posterior_count = train_posterior(results['theta'], x_flat_count, prior, cfg['method'])
    posterior_log10mass = train_posterior(results['theta'], x_flat_log10mass, prior, cfg['method'])
    posterior_count_Nmass = train_posterior(results['theta'], x_flat_count_Nmass, prior, cfg['method'])

    save_results(results, posterior_count_log10m, output_dir, 'count_log10mass'+cfg['output_name'] + '_')
    save_results(results, posterior_count, output_dir, 'count'+cfg['output_name'] + '_')
    save_results(results, posterior_log10mass, output_dir, 'log10mass'+cfg['output_name'] + '_')
    save_results(results, posterior_count_Nmass, output_dir, 'count_Nmass'+cfg['output_name'] + '_')

    print("\nCompleted successfully.")
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")

if __name__ == "__main__":
    main()