import numpy as np

analysis_list = []
analysis = dict()
analysis['filename'] = '../euclid_flagship_simulations/flagship_cluster_catalogue_summary_statstics_DES_MoR_no_log10mass_obs_scatter.npy'
analysis['Gamma'] = 0.7
analysis['summary_stat'] = 'count_only'
analysis['SSC_count_covariance'] = True
analysis['name_save'] = 'chains_count_only'

analysis1 = dict()
analysis1['filename'] = '../euclid_flagship_simulations/flagship_cluster_catalogue_summary_statstics_DES_MoR_no_log10mass_obs_scatter.npy'
analysis1['Gamma'] = 0.7
analysis1['summary_stat'] = 'mass_only'
analysis1['SSC_count_covariance'] = False
analysis1['name_save'] = 'chains_mass_only'

analysis2 = dict()
analysis2['filename'] = '../euclid_flagship_simulations/flagship_cluster_catalogue_summary_statstics_DES_MoR_no_log10mass_obs_scatter.npy'
analysis2['Gamma'] = 0.7
analysis2['summary_stat'] = 'count_mass'
analysis2['SSC_count_covariance'] = True
analysis2['name_save'] = 'chains_count_mass'

analysis_list = [analysis, analysis1, analysis2]