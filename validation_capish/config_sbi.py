import configparser
import copy
import io

def print_all_args(ini_file):

    for section in ini_file.sections():
        print(f"[{section}]")
        for key, value in ini_file[section].items():
            print(f"  {key} = {value}")
        print()
        
def clone_config(cfg):
    s = io.StringIO()
    cfg.write(s)
    s.seek(0)
    new_cfg = configparser.ConfigParser()
    new_cfg.read_file(s)
    return new_cfg

default_config_capish = configparser.ConfigParser()
default_config_capish.read('../config/capish_flagship.ini')
default_config_capish['cluster_catalogue']['theory_sigma_Mwl_gal'] = 'False'
default_config_capish['parameters']['sigma_Mwl_gal'] = '0.2'

########################################################################
# #######################################################################
# #######################################################################
n_replicate = 1
dparam = 0.00005
config_simulation = {"config.ini" : default_config_capish,
                    "config.ini_path" : None,
                    "output_name": "", 
                    "checkpoint_dir": "../../capish_sbi_data/config_sbi{}/checkpoint",
                    "output_dir": "../../capish_sbi_data/config_sbi{}/",
                    "variable_params_names" : ['Omega_m'],
                    "prior_min": [0.319 - dparam,],
                    "prior_max": [0.319 + dparam,],
                     "n_replicate" : 1,
                    "resume_from": None}
config_train ={"method": "NPE", 'summary_stat': ['count', 'Nm', 'count_Nm'], 'mask_empty_bins': False}
config_baseline = {"name"             : 'baseline_narrow_prior_1_param',
                   'config_simulation': config_simulation,
                   'config_train'     :config_train,}


config_simulation_1 = {"config.ini" : default_config_capish,
                       "config.ini_path" : None,
                       "output_name": "", 
                       "checkpoint_dir": "../../capish_sbi_data/config_sbi{}/checkpoint",
                       "output_dir": "../../capish_sbi_data/config_sbi{}/",
                       "variable_params_names" : ['Omega_m','sigma8'],
                       "prior_min"  : [0.2, 0.6],
                       "prior_max"  : [0.5, 0.9],
                       "n_replicate" : 1,
                       "resume_from": None}
config_train_1 ={"method": "NPE", 'summary_stat': ['count', 'Nm', 'count_Nm'], 'mask_empty_bins': False}
config_baseline_1 = {"name"            : 'baseline_standard_prior_2_params',
                    'config_simulation': config_simulation_1,
                    'config_train'     : config_train_1,}

config_simulation_2 = {"config.ini" : default_config_capish,
                       "config.ini_path" : None,
                       "output_name": "", 
                       "checkpoint_dir": "../../capish_sbi_data/config_sbi{}/checkpoint",
                       "output_dir": "../../capish_sbi_data/config_sbi{}/",
                       "variable_params_names" : ['Omega_m','sigma8','alpha_lambda' ,
                                               'beta_lambda', 'sigma_lambda'],
                       "prior_min"  : [0.2, 0.6, -12.0, 0.0, 0.0],
                       "prior_max"  : [0.5, 0.9,  -6.0, 2.0, 1.0],
                       "n_replicate" : 1,
                       "resume_from": None}
config_train_2 ={"method": "NPE", 'summary_stat': ['count', 'Nm', 'count_Nm'], 'mask_empty_bins': False}
config_baseline_2 = {"name"            : 'baseline_standard_prior_5_params',
                    'config_simulation': config_simulation_2,
                    'config_train'     : config_train_2,}


config_list = [config_baseline, config_baseline_1, config_baseline_2]

########################################################################
# #######################################################################
# ######################################################################
########################################################################
# #######################################################################
# #######################################################################

cfg_ini_new = clone_config(default_config_capish)
cfg_ini_new['cluster_catalogue.mass_observable_relation']['which_relation'] = 'power_law'
cfg_ini_new['cluster_catalogue']['gaussian_lensing_variable'] = 'log10Mwl'
cfg_ini_new['cluster_catalogue']['theory_sigma_Mwl_gal'] = 'True'
cfg_ini_new['summary_statistics']['use_stacked_sigma_Mwl_gal'] = 'True'
cfg_ini_new['summary_statistics']['use_stacked_sigma_Mwl_int'] = 'True'
cfg_ini_new['parameters']['sigma_Mwl_gal'] = '0.2'
cfg_ini_new['parameters']['sigma_Mwl_int'] = '0.0'
cfg_ini_new['parameters']['log10M_min'] = '14.5'
cfg_ini_new['parameters']['alpha_lambda'] = '3.5'
cfg_ini_new['parameters']['beta_lambda'] = '1.72'
cfg_ini_new['parameters']['gamma_lambda'] = '0.0'

new_config_list = []
for i in range(3):
    config_new = copy.deepcopy(config_list[i])
    config_new['config_simulation']['n_replicate'] = 1
    config_new['config_simulation']["config.ini"] = cfg_ini_new
    config_new['config_train'] ={"method": "NPE", 
                                 'summary_stat': ['count', 'Nm', 'count_Nm'], 
                                  #'summary_stat': ['count'], 
                                 'mask_empty_bins': False}
    if i==2:
        config_new['config_simulation']["variable_params_names"] = ['Omega_m','sigma8',
                                                                    'alpha_lambda','beta_lambda','gamma_lambda',
                                                                    'sigma_lambda']
        config_new['config_simulation']["prior_min"] = [0.2, 0.6, 
                                                        3.0, 1.3, 
                                                        -0.7, 0.1]
        config_new['config_simulation']["prior_max"] = [0.45, 0.95, 
                                                        4.0, 2.1, 
                                                        0.7, 0.5]
        config_new['name'] = 'DESlike6_corrected_standard_prior_6_params'
    else:
        config_new["name"] = 'DESlike6_corrected' + config_list[i]['name'].split('baseline')[1]
    new_config_list.append(config_new)


config_list = config_list + new_config_list

for i in range(len(config_list)):
    config_list[i]['config_simulation']['output_dir'] = config_list[i]['config_simulation']["output_dir"].format('_'+config_list[i]["name"])
    config_list[i]['config_simulation']['checkpoint_dir'] = config_list[i]['config_simulation']["checkpoint_dir"].format('_'+config_list[i]["name"])

config_dict = {config_list[i]['name']: config_list[i] for i in range(len(config_list))}

for k in config_dict.keys():

    print(k, config_dict[k]['config_simulation']["prior_max"])
    if k == 'DESlike3_MoR_log10Mwl_stacked_scatter_standard_prior_6_params':
        config_ini = config_dict[k]['config_simulation']["config.ini"]
        print(config_ini)
        print_all_args(config_ini)
        
    print()
    

#python sbi_train_posteriors.py --config_to_train DESlike6_corrected_standard_prior_6_params
#python sbi_sample_posteriors.py --config_to_sample DESlike6_corrected_standard_prior_6_params
#python sbi_test_posteriors.py --config_to_test DESlike6_corrected_standard_prior_6_params