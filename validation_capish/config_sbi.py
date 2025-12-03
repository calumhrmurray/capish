import configparser
import copy
import io

def clone_config(cfg):
    s = io.StringIO()
    cfg.write(s)
    s.seek(0)
    new_cfg = configparser.ConfigParser()
    new_cfg.read_file(s)
    return new_cfg

default_config_capish = configparser.ConfigParser()
default_config_capish.read('../config/capish_flagship.ini')

########################################################################
########################################################################
########################################################################

dparam = 0.00005
config_simulation = {"config.ini" : None,
                    "config.ini_path" : '../config/capish_flagship.ini',
                    "output_name": "", 
                    "checkpoint_dir": "./config_sbi{}/checkpoint",
                    "output_dir": "./config_sbi{}/",
                    "variable_params_names" : ['Omega_m'],
                    "prior_min": [0.319 - dparam,],
                    "prior_max": [0.319 + dparam,],
                    "resume_from": None}
config_train ={"method": "NPE"}
config_baseline = {"name"             : 'narrow_prior_1_param',
                   'config_simulation': config_simulation,
                   'config_train'     :config_train,}

########################################################################
########################################################################
########################################################################

config_simulation_1 = {"config.ini" : None,
                       "config.ini_path" : '../config/capish_flagship.ini',
                       "output_name": "", 
                       "checkpoint_dir": "./config_sbi{}/checkpoint",
                       "output_dir": "./config_sbi{}/",
                       "variable_params_names" : ['Omega_m','sigma8','alpha_lambda' ,
                                               'beta_lambda', 'sigma_lambda'],
                       "prior_min"  : [0.1, 0.3,-11.0, 0.0, 0.0],
                       "prior_max"  : [1.0, 1.0,-6.0,  2.0, 1.0],
                       "resume_from": None}

config_train_1 ={"method": "SNPE"}
config_baseline_1 = {"name"            : 'standard_prior_5_params',
                    'config_simulation': config_simulation_1,
                    'config_train'     : config_train_1,}

########################################################################
########################################################################
########################################################################

config_list = [config_baseline, config_baseline_1]

for i in range(len(config_list)):
    config_list[i]['config_simulation']['output_dir'] = config_list[i]['config_simulation']["output_dir"].format('_'+config_list[i]["name"])
    config_list[i]['config_simulation']['checkpoint_dir'] = config_list[i]['config_simulation']["checkpoint_dir"].format('_'+config_list[i]["name"])

config_dict = {config_list[i]['name']: config_list[i] for i in range(len(config_list))}

#python sbi_run_simulations.py --config_to_simulate narrow_prior_1_param --seed 30 --n_sims 20 --checkpoint_interval 10 --n_cores 3

#python sbi_train_posteriors.py --config_to_train narrow_prior_1_param