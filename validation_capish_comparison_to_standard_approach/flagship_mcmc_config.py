import numpy as np
import copy
analysis_list = []
analysis = dict()
analysis['filename'] = '../euclid_flagship_simulations/flagship_cluster_catalogue_summary_statstics_DES_MoR_no_log10mass_obs_scatter.npy'
analysis['Gamma'] = 0.7
analysis['summary_stat'] = 'count_only'
analysis['SSC_count_covariance'] = True
analysis['name_save'] = 'chains_count_only'
analysis['use_fiducial_data_vector'] = True

analysis1 = copy.deepcopy(analysis)
analysis1['summary_stat'] = 'mass_only'
analysis1['name_save'] = 'chains_mass_only'

analysis2 = copy.deepcopy(analysis)
analysis2['summary_stat'] = 'count_mass'
analysis2['name_save'] = 'chains_count_mass'

#########

analysis3 = copy.deepcopy(analysis)
analysis3['use_fiducial_data_vector'] = False
analysis3['name_save'] = 'chains_count_only_flagship'

analysis4 = copy.deepcopy(analysis1)
analysis4['use_fiducial_data_vector'] = False
analysis4['name_save'] = 'chains_mass_only_flagship'

analysis5 = copy.deepcopy(analysis2)
analysis5['use_fiducial_data_vector'] = False
analysis5['name_save'] = 'chains_count_mass_flagship'

analysis_list = [analysis, analysis1, analysis2]
analysis_list += [analysis3, analysis4, analysis5]