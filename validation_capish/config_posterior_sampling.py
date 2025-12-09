import configparser
import copy
import io, os
import torch

import pickle
import numpy as np
def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
    
def load_pickle(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin,  )

data = load_pickle('../../capish_sbi_data/config_sbi_baseline_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)

config_sampling_1 = {"name":'baseline_standard_prior_5_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'baseline_standard_prior_5_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

data = load_pickle('../../capish_sbi_data/config_sbi_theory_sigma_Mwl_gal_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)

config_sampling_2 = {"name":'theory_sigma_Mwl_gal_standard_prior_5_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'theory_sigma_Mwl_gal_standard_prior_5_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

data = load_pickle('../../capish_sbi_data/config_sbi_power_law_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)

config_sampling_3 = {"name":'power_law_standard_prior_5_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'power_law_standard_prior_5_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

data = load_pickle('../../capish_sbi_data/config_sbi_power_law_log10Mwl_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)

config_sampling_3 = {"name":'power_law_log10Mwl_standard_prior_5_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'power_law_log10Mwl_standard_prior_5_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

config_list=[config_sampling_1,config_sampling_2,config_sampling_3]

config_dict = {config['name']: config for config in config_list}

#python sbi_sample_posteriors --config_to_sample 