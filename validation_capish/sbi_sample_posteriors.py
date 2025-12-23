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
    summary = config_sbi_file['config_train']['summary_stat']

    save_dir = f'../../capish_sbi_data/config_sbi_{config_sbi_file["name"]}/'
    load_dir = f'../../capish_sbi_data/config_sbi_{config_sbi_file["name"]}/'

    os.system('mkdir ' + save_dir)

    ##########################################
    count = config_sampling["data_vector_count"].reshape(-1)
    log10m = config_sampling["data_vector_log10mass"].reshape(-1)
    Nm = count * 10 ** log10m
    count_log10m = np.concatenate([count, log10m])
    count_Nm = np.concatenate([count, Nm])

    obs_list_full = {'count': torch.tensor(count, dtype=torch.float32),
                'log10m': torch.tensor(log10m, dtype=torch.float32),
                'Nm': torch.tensor(Nm, dtype=torch.float32),
                'count_log10m': torch.tensor(count_log10m, dtype=torch.float32),
                'count_Nm': torch.tensor(count_Nm, dtype=torch.float32)}

    for k in obs_list_full.keys():

        label_ = k
        obs_ = obs_list_full[k]

        if label_ not in summary: continue
        masked ='_masked' if config_sbi_file['config_train']['mask_empty_bins'] else ''
        post_ =load_pickle(load_dir + f'{label_}{masked}_trained_posterior.pkl')
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
        save_pickle(posterior_samples, save_dir + 'samples_of_' + label_ + f'{masked}_posterior_with_data_'+ data_vector_infos+".pkl")
   
    return None

if __name__ == "__main__":
    sample_posterior()