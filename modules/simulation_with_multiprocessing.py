import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
import modules.simulation as simulation
import pyccl as ccl
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import numpy as np
import pickle

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
    
def load(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

def f_to_map(n):
    np.random.seed(n)
    print(f"Starting simulation {n}", flush=True)
    return simulator.run_simulation([0.25, 0.8])

def map(func, iterable, ncores=3):
    with Pool(processes=ncores) as pool:
        results = list(tqdm(pool.imap(func, iterable), total=len(iterable), desc="# progress ..."))
    return results

if __name__ == "__main__":
    simulator = simulation.UniverseSimulator(
                                default_config_path='../../config/capish.ini', 
                                variable_params_names=['Omega_m', 'sigma_8'])
    results = map(f_to_map, np.arange(30), ncores=10)
    save_pickle(results, './simulations_with_multiprocessing.pkl')
    