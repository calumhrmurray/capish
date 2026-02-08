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

config_sampling_7= {"name":'DESlike6_corrected_standard_prior_6_params',
                     "data_vector_infos": "flagship_like_sim",
                     "config_sbi" : 'DESlike6_corrected_standard_prior_6_params'}


config_list = [config_sampling_7]
config_dict = {config['name']: config for config in config_list}
#python sbi_test_posteriors.py --config_to_test DESlike6_corrected_standard_prior_6_params
