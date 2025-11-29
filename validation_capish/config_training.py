import configparser
import copy
import io

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
#alpha_mwl = 0.
#beta_mwl = 1.0
#gamma_mwl = 0.
#sigma_Mwl_gal = 0.0
#sigma_Mwl_int = 0.05

def clone_config(cfg):
    s = io.StringIO()
    cfg.write(s)
    s.seek(0)
    new_cfg = configparser.ConfigParser()
    new_cfg.read_file(s)
    return new_cfg
dparam = 0.001
default_config_capish = configparser.ConfigParser()
default_config_capish.read('../config/capish_internal_validation.ini')
config_1 = {"config.ini" :  None,
            "config.ini_path" :  '../config/capish_flagship.ini',
            "output_name": "",
            "method": "SNPE",
            "n_sims": 200,
            "n_cores": 4,
            "checkpoint_interval": 10,         
            "seed": 1,
            "checkpoint_dir": "./config_1/checkpoint/",
            "output_dir": "./config_1/",
            "variable_params_names" : ['Omega_m', 'sigma8', 'alpha_lambda', 'beta_lambda', 'sigma_lambda'],
            "prior_min": [0.319 - dparam, 0.813 - dparam, -9.348 - dparam, 0.75 - dparam, 0.3 - dparam],
            "prior_max": [0.319 + dparam, 0.813 + dparam, -9.348 + dparam, 0.75 + dparam, 0.3 + dparam],
            "resume_from": None}

config_list = [config_1]
