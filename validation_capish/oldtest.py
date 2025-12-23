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
    cfg_train = config_sbi_file['config_train']
    summary = cfg_train['summary_stat']
    masked= '_masked' if cfg_train['mask_empty_bins'] else ''

    save_dir = f'../../capish_sbi_data/config_sbi_{config_sbi_file["name"]}/'
    load_dir = f'../../capish_sbi_data/config_sbi_{config_sbi_file["name"]}/'

    #os.system('mkdir ' + save_dir)

    #load posteriors##########################################
    post ={s: load_pickle(load_dir + f'{s}{masked}_trained_posterior.pkl') for s in summary}
    print(post.keys())

    theta_fid = config_sampling["theta_fid"]
    ##########################################

    for key_post_ in post.keys():

        post_ = post[key_post_]
        print("#####################")
        print("test case =", key_post_)
        print()
    
        posterior_samples_ = []
        ntest = 300
        num_samples = 1000

        for i in range(ntest):
            count = config_sampling["data_vector_count"][i].reshape(-1)
            log10m = config_sampling["data_vector_log10mass"][i].reshape(-1)
            Nm = count * 10 ** log10m
            count_Nm = np.concatenate([count, Nm])
            count_log10m = np.concatenate([count, log10m])
    
            obs_list_i = {'count': count, 
                          'Nm': Nm, 
                          'log10m': log10m,
                          'count_Nm': count_Nm,
                          'count_log10m': count_log10m}
            
            obs_torch = {k: torch.tensor(obs_list_i[k], dtype=torch.float32) for k in obs_list_i.keys()}
    
            #choose observation matching the posterior
            obs_ = obs_torch[key_post_]
    
            posterior_samples = post_.sample((num_samples,), x=obs_).cpu().numpy()
            posterior_samples_.append(posterior_samples)
    
        alphas = np.linspace(0., 1, 30)
        num_params = len(theta_fid)
        coverages = np.zeros((len(alphas), num_params))
        
        for i, alpha in enumerate(alphas):
            for p in range(num_params):
                covered = 0
                for k in range(ntest):
                    lower = np.percentile(
                        posterior_samples_[k][:, p],
                        (1 - alpha) / 2 * 100)
                    upper = np.percentile(
                        posterior_samples_[k][:, p],
                        (1 + alpha) / 2 * 100)

                    if theta_fid[p] >= lower and theta_fid[p] <= upper:
                        covered += 1
                coverages[i, p] = covered / ntest
        
        # ---- Plot ----
        p_label = [r'$\Omega_m$',r'$\sigma_8$',r'$\alpha_\lambda$',r'$\beta_\lambda$', r'$\gamma_\lambda$', '$\sigma_\lambda$' ]
        plt.figure(figsize=(3,3))
        for p in range(num_params):
            plt.plot(alphas, coverages[:, p],'--', label=p_label[p])
        plt.plot([0,1], [0,1], 'k--', label='Ideal')
        plt.xlabel('Nominal confidence level (alpha)')
        plt.ylabel('Empirical coverage')
        plt.title('Coverage plot for all parameters')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"coverage_plot_{key_post_}.png", dpi=300, bbox_inches='tight')
        
    return None

if __name__ == "__main__":
    test_posterior()
