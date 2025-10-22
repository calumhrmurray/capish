import pickle
import glob
import sys, copy
from astropy.table import Table
import numpy as np
import pyccl as ccl
sys.path.append('/pbs/throng/lsst/users/cpayerne/capish/')
import modules.simulation as simulation
import matplotlib.pyplot as plt
def load(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)
import itertools

import configparser
default_config = configparser.ConfigParser()
default_config.read('../config/capish.ini')

simulator = simulation.UniverseSimulator(
                            default_config=default_config, 
                            variable_params_names=['Omega_m'])
Omegam_fid = [float(default_config['parameters']['Omega_m'])]
halo_cat_flagship = Table.read('../../capish_data/flagship.fits')
z = halo_cat_flagship['true_redshift_gal']
m200bh, m200ch, m500ch = halo_cat_flagship['m200b'], halo_cat_flagship['m200c'],halo_cat_flagship['m500c']
m200b, m200c, m500c = m200bh / 0.67, m200ch / 0.67, m500ch / 0.67

count200b, mean_log10m200b = simulator.run_simulation_from_halo_properties(np.log10(m200b), z, [Omegam_fid])
count200c, mean_log10m200c = simulator.run_simulation_from_halo_properties(np.log10(m200c), z, [Omegam_fid])

t = dict()
t['count_with_m200b_def'] = count200b
t['mean_log10m200b'] = mean_log10m200b
t['count_with_m200c_def'] = count200c
t['mean_log10m200c'] = mean_log10m200c

np.save('flagship_cluster_catalogue_summary_statstics_DES_MoR', t)