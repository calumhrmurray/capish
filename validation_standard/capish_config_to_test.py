import configparser
import copy
default_config_capish = configparser.ConfigParser()
default_config_capish.read('../config/capish.ini')
default_config_capish['cluster_catalogue']['add_completeness'] = 'False'
default_config_capish['cluster_catalogue']['add_purity'] = 'False'
default_config_capish['summary_statistics']['richness_edges'] = '20, 25, 30, 35, 40, 50, 80, 100'
default_config_capish['summary_statistics']['redshift_edges'] = '0.2, 0.5'

config_1 = {}
config_1['ini_file'] = copy.deepcopy(default_config_capish)
config_1['name'] = 'default_capish'

config_2 = {}
config_2['ini_file'] = copy.deepcopy(default_config_capish)
config_2['ini_file']['cluster_catalogue.mass_observable_relation']['which_relation'] = 'Gauss'
config_2['name'] = 'Gauss-only_MoR'

config_3 = {}
config_3['ini_file'] = copy.deepcopy(default_config_capish)
config_3['ini_file']['cluster_catalogue']['add_photometric_redshift'] = 'True'
config_3['name'] = 'with_photoz'

config_4 = {}
config_4['ini_file'] = copy.deepcopy(default_config_capish)
config_4['ini_file']['cluster_catalogue']['add_completeness'] = 'True'
config_4['name'] = 'incomplete'

config_4bis = {}
config_4bis['ini_file'] = copy.deepcopy(default_config_capish)
config_4bis['ini_file']['cluster_catalogue']['add_purity'] = 'True'
config_4bis['name'] = 'not_pure'

config_4bisbis = {}
config_4bisbis['ini_file'] = copy.deepcopy(default_config_capish)
config_4bisbis['ini_file']['cluster_catalogue']['add_purity'] = 'True'
config_4bisbis['ini_file']['cluster_catalogue']['add_completeness'] = 'True'
config_4bisbis['name'] = 'not_pure_and_incomplete'

config_5 = {}
config_5['ini_file'] = copy.deepcopy(default_config_capish)
config_5['ini_file']['parameters']['rho'] = '-0.1'
config_5['name'] = 'rho_m0.1'

config_5bis = {}
config_5bis['ini_file'] = copy.deepcopy(default_config_capish)
config_5bis['ini_file']['parameters']['rho'] = '0.1'
config_5bis['name'] = 'rho_p0.1'

config_5bisbis = {}
config_5bisbis['ini_file'] = copy.deepcopy(default_config_capish)
config_5bisbis['ini_file']['parameters']['rho'] = '0.2'
config_5bisbis['name'] = 'rho_p0.2'

config = [config_1,config_2,config_3,
          config_4,config_4bis,config_4bisbis,
          config_5bis,config_5bis,config_5bisbis]