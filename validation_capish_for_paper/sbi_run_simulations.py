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

import config_sbi

sys.path.append('../')
from modules.simulation_parallel import ParallelSimulator

def parse_args():

    parser = argparse.ArgumentParser(description='Run parallel SBI for Capish')
    parser.add_argument('--config_to_simulate', type=str, default='', help='which to train ?')
    parser.add_argument('--seed', type=int, default=20, help='seed')
    parser.add_argument('--n_sims', type=int, default=20, help='number of simulations')
    parser.add_argument('--n_cores', type=int, default=20, help='Number of CPU cores to use (default: 20)')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='checkpoint_interval')

    return parser.parse_args()

def create_simulator(var_names, config_path, default_config):

    return ParallelSimulator(default_config=default_config,
                             config_path=config_path,
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

def run_simulations(sim, theta, n_cores, ckpt_dir, checkpoint_interval):
    ckpt_path = ckpt_dir / f"checkpoint.pkl"
    print(f"\nRunning {len(theta)} parallel simulations on {n_cores} cores.")

    return sim.run_batch_parallel(
        theta_batch=theta,
        n_cores=n_cores,
        checkpoint_path=ckpt_path,
        checkpoint_interval=checkpoint_interval,
        desc="Simulations")

def save_results(results, out_dir, save_simulations=True, save_summary=True):
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_simulations:
        with open(out_dir / f"simulations.pkl", "wb") as f:
            pickle.dump(results, f)

    if save_summary:
        with open(out_dir / f"summary_of_simulations.txt", "w") as f:
            f.write("Capish SBI Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total simulations: {results['n_total']}\n")
            f.write(f"Successful: {results['n_success']}\n")
            f.write(f"Failed: {results['n_failed']}\n")
            f.write(f"Success rate: {results['success_rate']:.2%}\n")
            f.write(f"Elapsed time: {results['elapsed_time']:.1f}s\n")

def main():

    args = parse_args()
    name = args.config_to_simulate
    cfg_sims = config_sbi.config_dict[name]['config_simulation']
    n_sims = args.n_sims
    ncores = int(args.n_cores)
    seed = int(args.seed)
    checkpoint_interval = int(args.checkpoint_interval)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_dir = Path(cfg_sims['checkpoint_dir'])
    output_dir = Path(cfg_sims['output_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_replicate = cfg_sims['n_replicate']
    simulator = create_simulator(
        cfg_sims['variable_params_names'],
        cfg_sims["config.ini_path"],
        cfg_sims["config.ini"])
    
    prior = define_prior(cfg_sims['prior_min'], cfg_sims['prior_max'])
    if n_replicate == 1:
        theta = sample_parameters(prior, n_sims, seed)
    else:
        assert n_sims % n_replicate == 0
        n_theta_unique = n_sims // n_replicate
        theta_unique = sample_parameters(prior, n_theta_unique, seed)
        theta = np.repeat(theta_unique, n_replicate, axis=0)
    print(theta.shape)
    results = run_simulations(simulator, theta, ncores, checkpoint_dir, checkpoint_interval)

    print("Simulation Summary")
    print("=" * 70)
    print(f"Total: {results['n_total']}")
    print(f"Success: {results['n_success']}")
    print(f"Failed: {results['n_failed']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Time: {results['elapsed_time']:.1f}s")
    print("=" * 70)

    save_results(results, output_dir)

# Resume or run fresh
#    if cfg_sims['resume_from']:
#        ckpt = ParallelSimulator.load_checkpoint(cfg_sims['resume_from'])
#        results = ckpt['results']

#        remaining = n_sims - ckpt['n_completed']
#        if remaining > 0:
#            theta = sample_parameters(prior, remaining, seed + ckpt['n_completed'])
#            new = run_simulations(simulator, theta, ncores, checkpoint_dir, checkpoint_interval)

#            results['theta'] = np.vstack([results['theta'], new['theta']])
#            results['failed_theta'] = np.vstack([results['failed_theta'], new['failed_theta']])

#            if isinstance(results['x'], tuple):
#                results['x'] = tuple(
#                    np.vstack([results['x'][i], new['x'][i]])
#                    for i in range(len(results['x'])))
#            else:
#                results['x'] = np.vstack([results['x'], new['x']])
#
#            for k in ['n_total', 'n_success', 'n_failed', 'elapsed_time']:
#                results[k] += new[k]
#
#            results['success_rate'] = results['n_success'] / results['n_total']

#    else:

if __name__ == "__main__":
    main()