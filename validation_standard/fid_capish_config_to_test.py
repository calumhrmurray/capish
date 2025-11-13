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
default_config_capish.read('../config/capish_internal_validation.ini')
default_config_capish['cluster_catalogue']['add_completeness'] = 'False'
default_config_capish['cluster_catalogue']['add_purity'] = 'False'
default_config_capish['summary_statistics']['richness_edges'] = '20, 25, 30, 35, 40, 50, 80, 100'
default_config_capish['summary_statistics']['redshift_edges'] = '0.2, 0.5'

config_1 = {}
config_1['ini_file'] = clone_config(default_config_capish)
config_1['ini_file']['summary_statistics']['Gamma'] = '1.0'
config_1['ini_file']['parameters']['sigma_Mwl_gal'] = '0.0'
config_1['name'] = 'default_capish_Gamma_1.0_noWL_noise'

config_2 = {}
config_2['ini_file'] = clone_config(default_config_capish)
config_2['ini_file']['parameters']['sigma_Mwl_gal'] = '0.0'
config_2['name'] = 'default_capish_Gamma_0.7_noWL_noise'

config_2bis = {}
config_2bis['ini_file'] = clone_config(default_config_capish)
config_2bis['ini_file']['parameters']['sigma_Mwl_gal'] = '0.3'
config_2bis['name'] = 'default_capish_Gamma_0.7_WL_noise'

config_3 = {}
config_3['ini_file'] = clone_config(default_config_capish)
config_3['ini_file']['parameters']['sigma_Mwl_gal'] = '0.3'
config_3['ini_file']['cluster_catalogue']['add_photometric_redshift'] = 'True'
config_3['name'] = 'with_photoz'

config_4 = {}
config_4['ini_file'] = clone_config(default_config_capish)
config_4['ini_file']['parameters']['sigma_Mwl_gal'] = '0.3'
config_4['ini_file']['cluster_catalogue']['add_completeness'] = 'True'
config_4['name'] = 'incomplete'

config_4bis = {}
config_4bis['ini_file'] = clone_config(default_config_capish)
config_4bis['ini_file']['parameters']['sigma_Mwl_gal'] = '0.3'
config_4bis['ini_file']['cluster_catalogue']['add_purity'] = 'True'
config_4bis['name'] = 'not_pure'

config_4bisbis = {}
config_4bisbis['ini_file'] = clone_config(default_config_capish)
config_4bisbis['ini_file']['parameters']['sigma_Mwl_gal'] = '0.3'
config_4bisbis['ini_file']['cluster_catalogue']['add_purity'] = 'True'
config_4bisbis['ini_file']['cluster_catalogue']['add_completeness'] = 'True'
config_4bisbis['name'] = 'not_pure_and_incomplete'

config_5 = {}
config_5['ini_file'] = clone_config(default_config_capish)
config_5['ini_file']['parameters']['sigma_Mwl_gal'] = '0.3'
config_5['ini_file']['parameters']['rho'] = '-0.1'
config_5['name'] = 'rho_m0.1'

config_5bis = {}
config_5bis['ini_file'] = clone_config(default_config_capish)
config_5bis['ini_file']['parameters']['sigma_Mwl_gal'] = '0.3'
config_5bis['ini_file']['parameters']['rho'] = '0.1'
config_5bis['name'] = 'rho_p0.1'

config_5bisbis = {}
config_5bisbis['ini_file'] = clone_config(default_config_capish)
config_5bisbis['ini_file']['parameters']['sigma_Mwl_gal'] = '0.3'
config_5bisbis['ini_file']['parameters']['rho'] = '0.2'
config_5bisbis['name'] = 'rho_p0.2'

config = [config_1, config_2, config_2bis, config_3,
           config_4,config_4bis,config_4bisbis,
           config_5,config_5bis,config_5bisbis]

print(config_1['ini_file']['parameters']['sigma_Mwl_gal'])