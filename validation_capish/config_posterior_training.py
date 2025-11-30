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

dparam = 0.001
config_1 = {"name": 'narrow_prior_6_params',
            "config.ini" : None,
            "config.ini_path" :  '../config/capish_flagship.ini',
            "output_name": "",
            "method": "SNPE",      
            "checkpoint_dir": "./posterior_training{}/checkpoint",
            "output_dir": "./posterior_training{}/",
            "variable_params_names" : ['Omega_m', 'sigma8', 'alpha_lambda', 'beta_lambda', 'sigma_lambda'],
            "prior_min": [0.319 - dparam, 0.813 - dparam, -9.348 - dparam, 0.75 - dparam, 0.3 - dparam],
            "prior_max": [0.319 + dparam, 0.813 + dparam, -9.348 + dparam, 0.75 + dparam, 0.3 + dparam],
            "resume_from": None}

config_2 = {"name": 'standard_prior_6_params',
            "config.ini" : None,
            "config.ini_path" :  '../config/capish_flagship.ini',
            "output_name": "",
            "method": "SNPE",
            "checkpoint_dir": "./posterior_training{}/checkpoint",
            "output_dir": "./posterior_training{}/",
            "variable_params_names" : ['Omega_m', 'sigma8', 'alpha_lambda', 'beta_lambda', 'sigma_lambda'],
            "prior_min": [0.2, 0.5, -10, 0.0, 0.001],
            "prior_max": [0.6, 1.0, -7, 2.0, 1],
            "resume_from": None}

config_list = [config_1,config_2]


for i in range(len(config_list)):
    config_list[i]["output_dir"] = config_list[i]["output_dir"].format('_'+config_list[i]["name"])
    config_list[i]["checkpoint_dir"] = config_list[i]["checkpoint_dir"].format('_'+config_list[i]["name"])

config_dict = {config_list[i]['name']: config_list[i] for i in range(len(config_list))}