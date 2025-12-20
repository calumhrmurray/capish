import configparser
import copy
import io, os
import torch
import config_test_posterior
import config_sbi
import configparser
import pickle
import matplotlib.pyplot as plt
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
    parser.add_argument('--config_to_test', type=str)

    return parser.parse_args()

def test_posterior():

    args = parse_args()
    name = args.config_to_test

    config_sampling = config_test_posterior.config_dict[name]
    data_vector_infos = config_sampling["data_vector_infos"]
    config_sbi_file = config_sbi.config_dict[config_sampling['config_sbi']]

    save_dir = f'../../capish_sbi_data/config_sbi_{config_sbi_file["name"]}/'
    load_dir = f'../../capish_sbi_data/config_sbi_{config_sbi_file["name"]}/'

    #os.system('mkdir ' + save_dir)

    #load posteriors##########################################
    posterior_count = load_pickle(load_dir + 'count_trained_posterior.pkl')
    posterior_Nmass = load_pickle(load_dir + 'Nmass_trained_posterior.pkl')
    posterior_count_Nmass = load_pickle(load_dir + 'count_Nmass_trained_posterior.pkl')

    theta_fid = config_sampling["theta_fid"]
    print(theta_fid)
    
    post = [posterior_count,
            posterior_Nmass, 
            posterior_count_Nmass
           ]
    
    label =  ['count', 
              'Nm', 
              'count_Nm']

    ##########################################

    for post_, label_ in zip(post, label):
        print("#####################")
        print("test case =", label_)
        print()
    
        posterior_samples_ = []
        ntest = 100
        num_samples = 10000

        for i in range(ntest):
            count = config_sampling["data_vector_count"][i].reshape(-1)
            log10mass = config_sampling["data_vector_log10mass"][i].reshape(-1)
    
            Nmass = count * 10 ** log10mass
            count_Nmass = np.concatenate([count, Nmass])
    
            obs_list_i = [count, Nmass, count_Nmass]
            obs_torch = [torch.tensor(obs_j, dtype=torch.float32) for obs_j in obs_list_i]
    
            # choose observation matching the posterior
            if label_ == "count":
                obs_ = obs_torch[0]
            elif label_ == "Nm":
                obs_ = obs_torch[1]
            else:
                obs_ = obs_torch[2]
    
            posterior_samples = post_.sample((num_samples,), x=obs_).cpu().numpy()
            posterior_samples_.append(posterior_samples)
    
        alphas = np.linspace(0.1, 0.95, 10)
        num_params = len(theta_fid)
        coverages = np.zeros((len(alphas), num_params))
        
        for i, alpha in enumerate(alphas):
            for p in range(num_params):
                covered = 0
                for k in range(ntest):
                    lower = np.percentile(
                        posterior_samples_[k][:, p],
                        (1 - alpha) / 2 * 100
                    )
                    upper = np.percentile(
                        posterior_samples_[k][:, p],
                        (1 + alpha) / 2 * 100
                    )
                    if theta_fid[p] >= lower and theta_fid[p] <= upper:
                        covered += 1
                coverages[i, p] = covered / ntest
        
        # ---- Plot ----
        plt.figure(figsize=(3,3))
        for p in range(num_params):
            plt.plot(alphas, coverages[:, p], marker='o', label=f'param {p}')
        plt.plot([0,1], [0,1], 'k--', label='Ideal')
        plt.xlabel('Nominal confidence level (alpha)')
        plt.ylabel('Empirical coverage')
        plt.title('Coverage plot for all parameters')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"coverage_plot_{label_}.png", dpi=300, bbox_inches='tight')
        
    return None

if __name__ == "__main__":
    test_posterior()