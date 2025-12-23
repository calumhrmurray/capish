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

data = load_pickle('../../capish_sbi_data/config_sbi_DESlike6_corrected_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)


config_sampling= {"name":'DESlike6_corrected_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike6_corrected_standard_prior_6_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

data = load_pickle('../../capish_sbi_data/config_sbi_DESlike6_corrected_narrow_prior_1_param/simulations.pkl')
count =  data['x'][0][10]
log10mass = data['x'][1][10]


config_sampling1= {"name":'DESlike6_corrected_standard_prior_6_params_one_sim',
                     "data_vector_infos": "flagship_like_sim_one",
                     "config_sbi" : 'DESlike6_corrected_standard_prior_6_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

config_list = [config_sampling,config_sampling1]
config_dict = {config['name']: config for config in config_list}
#python sbi_sample_posteriors.py --config_to_sample DESlike6_corrected_standard_prior_6_params_one_sim