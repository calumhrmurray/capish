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


data = load_pickle('../../capish_sbi_data/config_sbi_DESlike_MoR_log10Mwl_stacked_scatter_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)

config_sampling_3 = {"name":'DESlike_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

data = load_pickle('../../capish_sbi_data/config_sbi_DESlike2_MoR_log10Mwl_stacked_scatter_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)

config_sampling_4= {"name":'DESlike2_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike2_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

data = load_pickle('../../capish_sbi_data/config_sbi_DESlike3_MoR_log10Mwl_stacked_scatter_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)

config_sampling_5= {"name":'DESlike3_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike3_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

data = load_pickle('../../capish_sbi_data/config_sbi_DESlike4_MoR_log10Mwl_stacked_scatter_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)

config_sampling_6= {"name":'DESlike4_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike4_MoR_log10Mwl_stacked_scatter_standard_prior_6_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

data = load_pickle('../../capish_sbi_data/config_sbi_DESlike4_replicate_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)

config_sampling_7= {"name":'DESlike4_replicate_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike4_replicate_standard_prior_6_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

data = load_pickle('../../capish_sbi_data/config_sbi_DESlike5_replicate_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
log10mass = np.mean(data['x'][1], axis=0)


config_sampling_8= {"name":'DESlike5_replicate_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike5_replicate_standard_prior_6_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

config_list = [config_sampling_3, config_sampling_4,config_sampling_5,config_sampling_6, config_sampling_8]
config_dict = {config['name']: config for config in config_list}
#python sbi_sample_posteriors.py --config_to_sample DESlike4_replicate_standard_prior_6_params