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
import fid_capish_config_to_test as capish_config_to_test

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()

code, index = sys.argv

config_file = capish_config_to_test.config[int(index)]

def map(func, iterable, ncores=10):
    with Pool(processes=ncores) as pool:
        results = list(tqdm(pool.imap(func, iterable), 
                    total=len(iterable), desc="# progress ..."))
    return results

simulator = simulation.UniverseSimulator(
                            default_config=config_file['ini_file'], 
                            variable_params_names=['Omega_m'])

def f_to_map(n):
    print(f"Starting simulation" +str(n), flush=True)
    return simulator.run_simulation([float(config_file['ini_file']['parameters']['Omega_m'])])

#results = map(f_to_map, np.arange(30), ncores=10)
results = []
for i in range(200):
    res = f_to_map(i)
    print(res)
    results.append(res)
save_pickle(results, '/pbs/throng/lsst/users/cpayerne/capish/validation_standard/capish_sims_at_fiducial_cosmology/'+config_file['name']+'.pkl')