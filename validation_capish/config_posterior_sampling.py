import configparser
import copy
import io, os
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

    config_training = config_posterior_training.config_dict[config_sampling["config_training"]]
    data_vector_infos = config_sampling["data_vector_infos"]

    save_dir = f'./posterior_sampling_{config_training["name"]}_with_{data_vector_infos}/'
    load_dir = f'./posterior_training_{config_training["name"]}/'

    os.system('mkdir ' + save_dir)

    posterior_count = load_pickle(load_dir + 'count_posterior.pkl')
    posterior_mass = load_pickle(load_dir + 'mass_posterior.pkl')
    posterior_count_mass = load_pickle(load_dir + 'count_mass_posterior.pkl')

    count = config_sampling["data_vector_count"].reshape(-1)
    log10mass = config_sampling["data_vector_mass"].reshape(-1)
    count_log10mass = np.concatenate([count, log10mass])
    count_Nmass = np.concatenate([count, count * 10 ** log10mass])
    
    observed_data_tensor_count = torch.tensor(count, dtype=torch.float32)
    observed_data_tensor_log10mass = torch.tensor(log10mass, dtype=torch.float32)
    observed_data_tensor_count_log10mass = torch.tensor(count_log10mass, dtype=torch.float32)
    observed_data_tensor_count_Nmass = torch.tensor(count_Nmass, dtype=torch.float32)

    num_samples = 500000

    # ---- Sampling ----
    posterior_count_samples_np = posterior_count.sample((num_samples,), x=observed_data_tensor_count).cpu().numpy()
    posterior_log10mass_samples_np = posterior_log10mass.sample((num_samples,), x=observed_data_tensor_log10mass).cpu().numpy()
    posterior_count_log10mass_samples_np = posterior_count_log10mass.sample((num_samples,), x=observed_data_tensor_count_log10mass).cpu().numpy()
    posterior_count_Nmass_samples_np = posterior_count_Nmass.sample((num_samples,), x=observed_data_tensor_count_Nmass).cpu().numpy()

    print(np.mean(posterior_count_samples_np[100000:], axis=0))
    print(np.mean(posterior_log10mass_samples_np[100000:], axis=0))
    print(np.mean(posterior_count_log10mass_samples_np[10000:], axis=0))
    print(np.mean(posterior_count_Nmass_samples_np[10000:], axis=0))

    # ---- Saving ----
    save_pickle(posterior_count_samples_np, save_dir + "count_posterior_samples.pkl")
    save_pickle(posterior_mass_samples_np, save_dir + "mass_posterior_samples.pkl")
    save_pickle(posterior_count_log10mass_samples_np, save_dir + "count_log10mass_posterior_samples.pkl")
    save_pickle(posterior_count_Mmass_samples_np, save_dir + "count_Nmass_posterior_samples.pkl")

    return None

data = load_pickle('./posterior_training_narrow_prior_2_params/count_mass_simulations.pkl')
count =  np.mean(data['x'][0], axis=0)
mass = np.mean(data['x'][1], axis=0)

default_config_capish = configparser.ConfigParser()
default_config_capish.read('../config/capish_internal_validation.ini')
config_sampling_1 = {"data_vector_infos": "flagship_like_sim",
                     "config_training" : 'standard_prior_6_params',
                     "data_vector_count" :count,
                     "data_vector_mass" : mass,}

run_posterior(config_sampling_1)
