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

def create_sims(config, name):

    simulator = simulation.UniverseSimulator(
                                default_config=config, 
                                variable_params_names=['Omega_m'])
    Omegam_fid = [float(default_config['parameters']['Omega_m'])]
    halo_cat_flagship = Table.read('../../capish_data/flagship.fits')
    z = halo_cat_flagship['true_redshift_gal']
    m200bh, m200ch, m500ch = halo_cat_flagship['m200b'], halo_cat_flagship['m200c'],halo_cat_flagship['m500c']
    m200b, m200c, m500c = m200bh / 0.67, m200ch / 0.67, m500ch / 0.67
    
    mask_m200b = m200b > 10 ** 13.3
    count200b, mean_log10m200b = simulator.run_simulation_from_halo_properties(np.log10(m200b)[mask_m200b], z[mask_m200b], [Omegam_fid])
    mask_m200c = m200c > 10 ** 13.3
    count200c, mean_log10m200c = simulator.run_simulation_from_halo_properties(np.log10(m200c)[mask_m200c], z[mask_m200c], [Omegam_fid])
    
    t = dict()
    t['count_with_m200b_def'] = count200b
    t['mean_log10m200b'] = mean_log10m200b
    t['count_with_m200c_def'] = count200c
    t['mean_log10m200c'] = mean_log10m200c
    
    np.save(f'flagship_cluster_catalogue_summary_statstics_{name}', t)

default_config_1 = copy.deepcopy(default_config)
default_config_1['parameters']['sigma_Mwl_gal'] = '0'
default_config_1['parameters']['sigma_Mwl_int'] = '0'
default_config_1['cluster_catalogue']['theory_sigma_Mwl_gal'] = 'False'
default_config_1['summary_statistics']['Gamma'] = '1'
create_sims(default_config_1, 'DES_MoR_no_Mwl_scatter_Gamma1')

default_config_2 = copy.deepcopy(default_config)
default_config_2['parameters']['sigma_Mwl_gal'] = '0'
default_config_2['parameters']['sigma_Mwl_int'] = '0.05'
default_config_2['cluster_catalogue']['theory_sigma_Mwl_gal'] = 'True'
default_config_2['summary_statistics']['Gamma'] = '1'
create_sims(default_config_2, 'DES_MoR_Mwl_scatter_Gamma1')

default_config_3 = copy.deepcopy(default_config)
default_config_3['parameters']['sigma_Mwl_gal'] = '0'
default_config_3['parameters']['sigma_Mwl_int'] = '0.05'
default_config_3['cluster_catalogue']['theory_sigma_Mwl_gal'] = 'True'
default_config_3['summary_statistics']['Gamma'] = '0.7'
create_sims(default_config_3, 'DES_MoR_Mwl_scatter_Gamma0.7')

name = 'DES_MoR_no_log10mass_obs_scatter'