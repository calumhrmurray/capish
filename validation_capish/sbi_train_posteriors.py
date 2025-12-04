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

def define_prior(prior_min, prior_max):
    return BoxUniform(
        low=torch.tensor(prior_min),
        high=torch.tensor(prior_max))

def flatten_summary_stats(x):
    if isinstance(x, tuple):
        flattened = [xi.reshape(len(xi), -1) for xi in x]
        return np.concatenate(flattened, axis=1)
    return x.reshape(len(x), -1)

def train_posterior(theta, x, prior, method="SNPE"):
    print(f"\nTraining {method} posterior on {len(theta)} samples…")

    theta = torch.tensor(theta, dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)

    inference = SNPE(prior=prior)

    batch_size = min(50, max(10, len(theta) // 10))
    density_est = inference.append_simulations(theta, x).train(
                        training_batch_size=batch_size)

    return inference.build_posterior(density_est)

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
    
def load_pickle(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

def parse_args():

    parser = argparse.ArgumentParser(description='Run parallel SBI for Capish')
    parser.add_argument('--config_to_train', type=str, default='', help='which to train ?')

    return parser.parse_args()

def train():

    args = parse_args()
    name = args.config_to_train

    cfg_sims = config_sbi.config_dict[name]['config_simulation']
    cfg_train = config_sbi.config_dict[name]['config_train']

    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_dir = Path(cfg_sims['checkpoint_dir'])
    output_dir = Path(cfg_sims['output_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_name = cfg_sims["output_name"]
    
    results = load_pickle(output_dir / f"simulations.pkl")

    print("Simulation Summary")
    print("=" * 70)
    print(f"Total: {results['n_total']}")
    print(f"Success: {results['n_success']}")
    print(f"Failed: {results['n_failed']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Time: {results['elapsed_time']:.1f}s")
    print("=" * 70)

    #alone
    x_count      = results['x'][0]
    x_log10mass  = results['x'][1]
    x_Nlog10mass = results['x'][0] * results['x'][1]
    x_Nmass      = results['x'][0] * 10 ** results['x'][1]
    #together
    x_count_log10mass = (x_count, x_log10mass)
    x_count_Nlog10mass = (x_count, x_Nlog10mass)
    x_count_Nmass = (x_count, x_count * 10 ** results['x'][1])

    x = [x_count,x_log10mass, 
         x_Nlog10mass, x_Nmass, 
         x_count_log10mass, x_count_Nlog10mass, x_count_Nmass]
    
    x_label = ['count','log10mass', 
               'Nlog10mass', 'Nmass',
               'count_log10mass', 'count_Nlog10mass', 'count_Nmass']

    prior = define_prior(cfg_sims['prior_min'], cfg_sims['prior_max'])

    for x_, x_label_ in zip(x, x_label):
        x_flat_ = flatten_summary_stats(x_)
        name = cfg_sims['output_dir'] +  '/' + x_label_ + f"_trained_posterior.pkl"
        if os.path.exists(name): continue
        posterior_ = train_posterior(results['theta'], x_flat_, prior, cfg_train['method'])
        with open(name, "wb") as f:
            pickle.dump(posterior_, f)
   
    print("\nCompleted successfully.")
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")

if __name__ == "__main__":
    train()