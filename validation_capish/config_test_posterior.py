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


data = load_pickle('../../capish_sbi_data/config_sbi_DESlike4_MoR_log10Mwl_stacked_scatter_narrow_prior_1_param/simulations.pkl')
counts_fid =  data['x'][0]
log10masses_fid = data['x'][1]
theta_fid = [0.319, 0.813, 3.5, 1.72, 0, 0.2]

config_sampling_6= {"name":'DESlike4_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike4_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_count" :counts_fid,
                     "data_vector_log10mass" : log10masses_fid,
                     "theta_fid" : theta_fid}

data = load_pickle('../../capish_sbi_data/config_sbi_DESlike5_replicate_narrow_prior_1_param/simulations.pkl')
counts_fid =  data['x'][0]
log10masses_fid = data['x'][1]
theta_fid = [0.319, 0.813, 3.5, 1.72, 0, 0.2]

config_sampling_7= {"name":'DESlike5_replicate_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike5_replicate_standard_prior_6_params',
                     "data_vector_count" :counts_fid,
                     "data_vector_log10mass" : log10masses_fid,
                     "theta_fid" : theta_fid}


config_list = [config_sampling_6,config_sampling_7]
config_dict = {config['name']: config for config in config_list}
#python sbi_test_posteriors.py --config_to_test DESlike4_replicate_standard_prior_6_params
