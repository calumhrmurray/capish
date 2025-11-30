import configparser
import copy
import io
import torch
import config_posterior_training

import pickle
import numpy as np
def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
    
def load_pickle(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin,  )


def clone_config(cfg):
    s = io.StringIO()
    cfg.write(s)
    s.seek(0)
    new_cfg = configparser.ConfigParser()
    new_cfg.read_file(s)
    return new_cfg

def run_posterior(config_sampling):

    config_training = config_sampling["config_training"]

    save_dir = f'./posterior_sampling_{config_training["name"]}/'

    posterior_count = load_pickle(save_dir + 'count_posterior.pkl')
    posterior_mass = load_pickle(save_dir + 'mass_posterior.pkl')
    posterior_count_mass = load_pickle(save_dir + 'count_mass_posterior.pkl')

    count = config_sampling["data_vector_count"].reshape(-1)
    mass = config_sampling["data_vector_mass"].reshape(-1)
    count_mass = np.concatenate([count, mass])
    
    observed_data_tensor_count = torch.tensor(count, dtype=torch.float32)
    observed_data_tensor_mass = torch.tensor(mass, dtype=torch.float32)
    observed_data_tensor_count_mass = torch.tensor(count_mass, dtype=torch.float32)

    num_samples = 500000

    # ---- Sampling ----
    posterior_count_samples_np = posterior_count.sample((num_samples,), x=observed_data_tensor_count).cpu().numpy()
    posterior_mass_samples_np = posterior_mass.sample((num_samples,), x=observed_data_tensor_mass).cpu().numpy()
    posterior_count_mass_samples_np = posterior_count_mass.sample((num_samples,), x=observed_data_tensor_count_mass).cpu().numpy()

    # ---- Saving ----
    save_pickle(posterior_count_samples_np, save_dir + "count_posterior_samples.pkl")
    save_pickle(posterior_mass_samples_np, save_dir + "mass_posterior_samples.pkl")
    save_pickle(posterior_count_mass_samples_np, save_dir + "count_mass_posterior_samples.pkl")

    return None


data = load_pickle('./posterior_training_narrow_prior_6_params/count_mass_simulations.pkl')

default_config_capish = configparser.ConfigParser()
default_config_capish.read('../config/capish_internal_validation.ini')
config_sampling_1 = {"data_vector_infos": "flagship_like_sim",
                     "config_training" : config_posterior_training.config_dict['config_1'],
                     "data_vector_count" :data['x'][0][0],
                     "data_vector_mass" : data['x'][1][0],}

run_posterior(config_sampling_1)
