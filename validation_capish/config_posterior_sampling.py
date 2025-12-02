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
    posterior_log10mass = load_pickle(load_dir + 'log10mass_posterior.pkl')
    posterior_Nmass = load_pickle(load_dir + 'Nmass_posterior.pkl')
    posterior_count_log10mass = load_pickle(load_dir + 'count_log10mass_posterior.pkl')
    posterior_count_Nmass = load_pickle(load_dir + 'count_Nmass_posterior.pkl')

    count = config_sampling["data_vector_count"].reshape(-1)
    log10mass = config_sampling["data_vector_log10mass"].reshape(-1)
    Nmass = count * 10 ** log10mass
    count_log10mass = np.concatenate([count, log10mass])
    count_Nmass = np.concatenate([count, Nmass])
    
    observed_data_tensor_count = torch.tensor(count, dtype=torch.float32)
    observed_data_tensor_log10mass = torch.tensor(log10mass, dtype=torch.float32)
    observed_data_tensor_Nmass = torch.tensor(Nmass, dtype=torch.float32)
    observed_data_tensor_count_log10mass = torch.tensor(count_log10mass, dtype=torch.float32)
    observed_data_tensor_count_Nmass = torch.tensor(count_Nmass, dtype=torch.float32)

    obs = [observed_data_tensor_count, 
           observed_data_tensor_log10mass, 
           observed_data_tensor_Nmass, 
           observed_data_tensor_count_log10mass, 
           observed_data_tensor_count_Nmass]
    
    post = [posterior_count, 
            posterior_log10mass, 
            posterior_Nmass, 
            posterior_count_log10mass, 
            posterior_count_Nmass]
    
    label =  ['count', 
              'log10m', 
              'Nm', 
              'count_log10m', 
              'count_Nm']

    for obs_, post_, label_ in zip(obs, post, label):
        print("#####################")
        print()
        num_samples = 500000
        posterior_samples = post_.sample((num_samples,), x=obs_).cpu().numpy()
        param = np.mean(posterior_samples[400000:], axis=0)
        std = np.std(posterior_samples[400000:], axis=0)
        print('test case=', label_)
        print()
        params = config_training["variable_params_names"]
        for i, p in enumerate(params):
            print(p, ' : ', param[i], ' pm ', std[i])
        print()

        samples = posterior_samples[400000:]   # burn-in removed

    # Compute correlation matrix (params x params)
        corr = np.corrcoef(samples, rowvar=False)
        print("Parameter Correlations:")
        for i in range(len(params)):
            for j in range(len(params)):
                if j <= i: continue
                print(f"corr({params[i]}, {params[j]}) = {corr[i,j]:.4f}")
        print()
        print("#####################")

    # ---- Saving ----
        save_pickle(posterior_samples, save_dir + label_ + "_posterior_samples.pkl")
   

    return None

data = load_pickle('./posterior_training_narrow_prior_1_param/count_log10mass_simulations.pkl')
count =  np.mean(data['x'][0], axis=0)
log10mass = np.mean(data['x'][1], axis=0)

default_config_capish = configparser.ConfigParser()
default_config_capish.read('../config/capish_internal_validation.ini')
config_sampling_1 = {"data_vector_infos": "flagship_like_sim",
                     "config_training" : 'standard_prior_2_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}
config_sampling_2 = {"data_vector_infos": "flagship_like_sim",
                     "config_training" : 'standard_prior_5_params',
                     "data_vector_count" :count,
                     "data_vector_log10mass" : log10mass,}

run_posterior(config_sampling_2)