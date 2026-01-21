import configparser
import copy
import io, os
import torch
import config_test_posterior
import config_sbi
import configparser
import pickle
from sbi.utils import BoxUniform
from sbi.inference import SNPE
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

    sims = load_pickle(load_dir + f'simulations.pkl')
    count = sims['x'][0]
    log10m = sims['x'][1]
    theta = sims['theta']
    if cfg_train['mask_empty_bins']: 
        mask = []
        for i in range(len(count)):
            x = count[i]
            x = np.sum(x==0)
            if x >= 1: mask.append(False)
            else: mask.append(True)
        
        mask = np.array(mask)
    else: mask = np.zeros(len(count)) == 0
    restricted_center = True
    if restricted_center:
        Om_min, Om_max = 0.22, 0.43
        sigma8_min, sigma8_max = 0.64, 0.91
        a_min, a_max = 3.1, 3.9
        b_min, b_max = 1.32, 1.9
        c_min, c_max = -0.6, 0.6
        sigma_min, sigma_max = 0.11, 0.49
        mask *= (theta[:,0] > Om_min)*(theta[:,0] < Om_max)
        mask *= (theta[:,1] > sigma8_min)*(theta[:,1] < sigma8_max)
        mask *= (theta[:,2] > a_min)*(theta[:,2] < a_max)
        mask *= (theta[:,3] > b_min)*(theta[:,3] < b_max)
        mask *= (theta[:,4] > c_min)*(theta[:,4] < c_max)
        mask *= (theta[:,5] > sigma_min)*(theta[:,5] < sigma_max)
    count = count[mask]
    log10m = log10m[mask]
    ##########################################

    for key_post_ in post.keys():
        if key_post_ != 'count_Nm': continue

        post_ = post[key_post_]
        print("#####################")
        print("test case =", key_post_)
        print()
    
        posterior_samples_ = []
        ntest = 1000
        num_samples = 5000

        for i in range(ntest):
            print('test = ', i)
            count_i = count[i].reshape(-1)
            log10m_i = log10m[i].reshape(-1)
            Nm_i = count_i * 10 ** log10m_i
            count_Nm_i = np.concatenate([count_i, Nm_i])
            count_log10m_i = np.concatenate([count_i, log10m_i])
    
            obs_list_i = {'count': count_i, 
                          'Nm': Nm_i, 
                          'log10m': log10m_i,
                          'count_Nm': count_Nm_i,
                          'count_log10m': count_log10m_i}
            
            obs_torch = {k: torch.tensor(obs_list_i[k], dtype=torch.float32) for k in obs_list_i.keys()}
    
            #choose observation matching the posterior
            obs_ = obs_torch[key_post_]
    
            if restricted_center:

                def in_restricted_region(theta):
                    return (
                        (theta[:, 0] > Om_min) & (theta[:, 0] < Om_max) &
                        (theta[:, 1] > sigma8_min) & (theta[:, 1] < sigma8_max) &
                        (theta[:, 2] > a_min) & (theta[:, 2] < a_max) &
                        (theta[:, 3] > b_min) & (theta[:, 3] < b_max) &
                        (theta[:, 4] > c_min) & (theta[:, 4] < c_max) &
                        (theta[:, 5] > sigma_min) & (theta[:, 5] < sigma_max)
                    )
                def sample_posterior_restricted(post, obs, num_samples, batch_size=2000):
                    accepted = []
                    while sum(len(a) for a in accepted) < num_samples:
                        s = post.sample((batch_size,), x=obs).cpu().numpy()
                        m = in_restricted_region(s)
                        if np.any(m):
                            accepted.append(s[m])
                
                    samples = np.concatenate(accepted, axis=0)[:num_samples]
                    return samples

                posterior_samples = sample_posterior_restricted(post_, obs_, num_samples, batch_size=2000)
                
                #def mask_sample(posterior_sample):
                #    mask_res = (posterior_sample[:,0] > Om_min)*(posterior_sample[:,0] < Om_max)
                #    mask_res *= (posterior_sample[:,1] > sigma8_min)*(posterior_sample[:,1] < sigma8_max)
                #    mask_res *= (posterior_sample[:,2] > a_min)*(posterior_sample[:,2] < a_max)
                #    mask_res *= (posterior_sample[:,3] > b_min)*(posterior_sample[:,3] < b_max)
                #    mask_res *= (posterior_sample[:,4] > c_min)*(posterior_sample[:,4] < c_max)
                #    mask_res *= (posterior_sample[:,5] > sigma_min)*(posterior_sample[:,5] < sigma_max)
                #    return mask_res
                    
                #mask_res = mask_sample(posterior_samples)
                posterior_samples_.append(posterior_samples)#[mask_res])
            else: 
                posterior_samples = post_.sample((num_samples,), x=obs_).cpu().numpy()
                posterior_samples_.append(posterior_samples)

        if key_post_ == 'count_Nm':
            masked = '_masked' if cfg_train['mask_empty_bins'] else ''
            restricted = '_restricted_center' if restricted_center else ''
            theta_to_save = sims['theta'][mask][:ntest]
            dict_tarp = {'theta': theta_to_save, 'sample' : posterior_samples_}
            save_pickle(dict_tarp, f"{save_dir}/tarp_sample_{key_post_}{masked}{restricted}.pkl")
    
        alphas = np.linspace(0., 1, 30)
        num_params = 6
        coverages = np.zeros((len(alphas), num_params))
        
        for i, alpha in enumerate(alphas):
            for p in range(num_params):
                covered = 0
                for k in range(ntest):
                    theta_fid = sims['theta'][mask][k]
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
        p_label = [r'$\Omega_m$', r'$\sigma_8$', r'$\mu^\lambda_0$', r'$\mu^\lambda_m$',r'$\mu^\lambda_z$',r'$\sigma_{\ln \lambda, \rm int}$']
        
        plt.figure(figsize=(3,3))
        for p in range(num_params):
            plt.plot(alphas, coverages[:, p],'--', label=p_label[p])
        plt.plot([0,1], [0,1], 'k-', zorder=0)
        plt.xlabel(r'Confidence level ($\gamma$)')
        plt.ylabel(r'Probability coverage ($p_\gamma$)')
        plt.title(key_post_)
        plt.legend()
        #plt.grid(True)
        masked = '_masked' if cfg_train['mask_empty_bins'] else ''
        restricted = '_restricted_center' if restricted_center else ''
        #plt.savefig(f"diagnostic_coverage_plot_{key_post_}{masked}{restricted}.png", dpi=300, bbox_inches='tight')
        coverage_out = {
            "alphas": alphas,                 # (n_alpha,)
            "coverages": coverages,           # (n_alpha, n_params)
            "param_labels": p_label,          # list of strings
            "posterior_key": key_post_,}

        #save_pickle(coverage_out, f"{save_dir}/coverage_data_{key_post_}{masked}{restricted}.pkl")



        ####################
        ####################
        # ---- Compute posterior mean and bias ----
        theta_true_all = sims['theta'][mask][:ntest]  # (ntest, num_params)
        theta_mean_all = np.zeros_like(theta_true_all)

        for k in range(ntest):
            theta_mean_all[k] = posterior_samples_[k].mean(axis=0)

            # ---- Bias plot: <theta_hat - theta_true> vs theta_true ----
        bias_all = theta_mean_all - theta_true_all

        fig, axes = plt.subplots(2, 3, figsize=(9,5), sharex=False)
        axes = axes.flatten()
        
        for p in range(num_params):
            ax = axes[p]
        
            ax.scatter(
                theta_true_all[:, p],
                theta_mean_all[:, p],
                s=4,
                alpha=1, c='C0',
            )

            x = np.linspace(np.min(theta_true_all[:, p]), np.max(theta_true_all[:, p]), 10)
        
            ax.plot(x,x, color='r', ls='--', linewidth=3, zorder=1000)
        
            ax.set_xlabel(r"$\theta_{\rm true}$")
            ax.set_ylabel(r"$\hat{\theta}$")
            ax.set_title(p_label[p])
            #ax.grid(True)
        
        fig.suptitle("Bias per parameter – " + key_post_, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        #fig.savefig(f"diagnostic_bias_subplot_{key_post_}{masked}{restricted}.png",dpi=300,bbox_inches="tight")
        plt.close(fig)
        bias_out = {
            "theta_true": theta_true_all,      # (n_sims, n_params)
            "theta_mean": theta_mean_all,      # (n_sims, n_params)
            "param_labels": p_label,
            "posterior_key": key_post_,}

        #save_pickle(bias_out,f"{save_dir}/bias_data_{key_post_}{masked}{restricted}.pkl")


        
    return None

if __name__ == "__main__":
    test_posterior()