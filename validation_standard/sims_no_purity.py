
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/capish/')
import modules.simulation as simulation
import pyccl as ccl
import configparser
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import numpy as np
import pickle

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()

def f_to_map(n):
    np.random.seed(n)
    print(f"Starting simulation" +str(n), flush=True)
    return simulator.run_simulation([0.26, 0.8])

def map(func, iterable, ncores=3):
    with Pool(processes=ncores) as pool:
        results = list(tqdm(pool.imap(func, iterable), total=len(iterable), desc="# progress ..."))
    return results

if __name__ == "__main__":
    default_config_path = '/pbs/throng/lsst/users/cpayerne/capish/config/capish.ini'
    default_config = configparser.ConfigParser()
    default_config.read(default_config_path)
    default_config['cluster_catalogue']['add_purity'] = 'False'
    simulator = simulation.UniverseSimulator(
                                default_config=default_config, 
                                variable_params_names=['Omega_m', 'sigma_8'])
    
    results = map(f_to_map, np.arange(30), ncores=10)
    save_pickle(results, '/pbs/throng/lsst/users/cpayerne/capish/validation_standard/sims_no_purity.pkl')
            