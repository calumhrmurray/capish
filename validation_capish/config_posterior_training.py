import configparser
import copy
import io

#[#parameters]
#cosmology
#Omega_m = 0.319
#sigma8 = 0.813
#Omega_b = 0.048254
#ns = 0.96
#h = 0.67
#w0 = -1
#wa = 0
#mass-observable relation
##richness (lambda)
#M_min = 134896288259.1656
#alpha_lambda = -9.348
#beta_lambda = 0.75
#gamma_lambda = 0.0
#sigma_lambda = 0.3
##weak lensing mass (Mwl)

def clone_config(cfg):
    s = io.StringIO()
    cfg.write(s)
    s.seek(0)
    new_cfg = configparser.ConfigParser()
    new_cfg.read_file(s)
    return new_cfg

default_config_capish = configparser.ConfigParser()
default_config_capish.read('../config/capish_flagship.ini')

dparam = 0.00005
config_1 = {"name": 'narrow_prior_1_param',
            "config.ini" : None,
            "config.ini_path" : '../config/capish_flagship.ini',
            "output_name": "",
            "method": "SNPE",      
            "checkpoint_dir": "./posterior_training{}/checkpoint",
            "output_dir": "./posterior_training{}/",
            "variable_params_names" : ['Omega_m'],
            "prior_min": [0.319 - dparam,],
            "prior_max": [0.319 + dparam,],
            "resume_from": None}

#python run_sbi_parallel_from_config_posterior_training.py --config_to_train narrow_prior_1_param --seed 30 --n_sims 20 --checkpoint_interval 10 --n_cores 3

config_2 = {"name": 'standard_prior_2_params',
            "config.ini" : None,
            "config.ini_path" :  '../config/capish_flagship.ini',
            "output_name": "",
            "method": "SNPE",
            "checkpoint_dir": "./posterior_training{}/checkpoint",
            "output_dir": "./posterior_training{}/",
            "variable_params_names" : ['Omega_m', 'sigma8'],
            "prior_min": [0.1, 0.2,],
            "prior_max": [1.0, 1.0,],
            "resume_from": None}

#cosmology
#Omega_m = 0.319
#sigma8 = 0.813
#alpha_lambda = -9.348
#beta_lambda = 0.75
#gamma_lambda = 0.0
#sigma_lambda = 0.3

config_3 = {"name": 'standard_prior_5_params',
            "config.ini" : None,
            "config.ini_path" :  '../config/capish_flagship.ini',
            "output_name": "",
            "method": "SNPE",
            "checkpoint_dir": "./posterior_training{}/checkpoint",
            "output_dir": "./posterior_training{}/",
            "variable_params_names" : ['Omega_m','sigma8','alpha_lambda' ,'beta_lambda', 'sigma_lambda'],
            "prior_min": [0.1, 0.2,-13.0, 0.1, 0.01],
            "prior_max": [1.0, 1.0,-5.0,  2.0, 1.0],
            "resume_from": None}

config_list = [config_1,config_2,config_3]


for i in range(len(config_list)):
    config_list[i]["output_dir"] = config_list[i]["output_dir"].format('_'+config_list[i]["name"])
    config_list[i]["checkpoint_dir"] = config_list[i]["checkpoint_dir"].format('_'+config_list[i]["name"])

config_dict = {config_list[i]['name']: config_list[i] for i in range(len(config_list))}