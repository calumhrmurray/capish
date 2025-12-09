import configparser
import copy
import io, os
import torch
import config_posterior_sampling
import config_sbi
import configparser
import pickle
import argparse
import numpy as np
def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
    
def load_pickle(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin,  )

def parse_args():

    parser = argparse.ArgumentParser(description='Run parallel SBI for Capish')
    parser.add_argument('--config_to_sample', type=str, default='', help='which to train ?')

    return parser.parse_args()

def sample_posterior():

    args = parse_args()
    name = args.config_to_sample

    config_sampling = config_posterior_sampling.config_dict[name]
    data_vector_infos = config_sampling["data_vector_infos"]
    config_sbi_file = config_sbi.config_dict[config_sampling['config_sbi']]

    save_dir = f'../../capish_sbi_data/config_sbi_{config_sbi_file["name"]}/'
    load_dir = f'../../capish_sbi_data/config_sbi_{config_sbi_file["name"]}/'

    os.system('mkdir ' + save_dir)

    #load posteriors##########################################
    posterior_count = load_pickle(load_dir + 'count_trained_posterior.pkl')
    #posterior_log10mass = load_pickle(load_dir + 'log10mass_trained_posterior.pkl')
    #posterior_Nlog10mass = load_pickle(load_dir + 'Nlog10mass_trained_posterior.pkl')
    posterior_Nmass = load_pickle(load_dir + 'Nmass_trained_posterior.pkl')
    #
    #posterior_count_log10mass = load_pickle(load_dir + 'count_log10mass_trained_posterior.pkl')
    #posterior_count_Nlog10mass = load_pickle(load_dir + 'count_Nlog10mass_trained_posterior.pkl')
    posterior_count_Nmass = load_pickle(load_dir + 'count_Nmass_trained_posterior.pkl')

    ##########################################
    count = config_sampling["data_vector_count"].reshape(-1)
    log10mass = config_sampling["data_vector_log10mass"].reshape(-1)
    Nmass = count * 10 ** log10mass
    Nlog10mass = count * log10mass
    count_log10mass = np.concatenate([count, log10mass])
    count_Nlog10mass = np.concatenate([count, count*log10mass])
    count_Nmass = np.concatenate([count, count*10**log10mass])

    obs_list = [count, 
                #log10mass,
                #Nlog10mass,
                Nmass,
                #count_log10mass,
                #count_Nlog10mass, 
                count_Nmass
               ]
    obs_torch = [torch.tensor(obs_i, dtype=torch.float32) for obs_i in obs_list]
    
    
    post = [posterior_count,
            #posterior_log10mass, 
            #posterior_Nlog10mass, 
            posterior_Nmass, 
            #posterior_count_log10mass, 
            #posterior_count_Nlog10mass, 
            posterior_count_Nmass
           ]
    
    label =  ['count', 
              #'log10mass',
              #'Nlog10m',
              'Nm', 
              #'count_log10m',
              #'count_Nlog10m',
              'count_Nm']

    for obs_, post_, label_ in zip(obs_torch, post, label):
        print("#####################")
        print('test case=', label_)
        print()
        num_samples = 500000
        posterior_samples = post_.sample((num_samples,), x=obs_).cpu().numpy()
        param = np.mean(posterior_samples[400000:], axis=0)
        std = np.std(posterior_samples[400000:], axis=0)
        print()
        params = config_sbi_file['config_simulation']["variable_params_names"]
        print(params)
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
        save_pickle(posterior_samples, save_dir + 'samples_of_' + label_ + '_posterior_with_data_'+ data_vector_infos+".pkl")
   
    return None

if __name__ == "__main__":
    sample_posterior()