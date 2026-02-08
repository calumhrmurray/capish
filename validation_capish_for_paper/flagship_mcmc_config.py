import numpy as np
import copy


import pickle
import numpy as np
def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
    
def load_pickle(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin,  )

data = load_pickle('../../capish_sbi_data/config_sbi_DESlike6_corrected_narrow_prior_1_param/simulations.pkl')
count =  np.mean(data['x'][0], axis=0) 
n_real, n_r, n_z = data['x'][0].shape
counts_flat = data['x'][0].reshape(n_real, n_r * n_z)
cov_count = np.cov(counts_flat, rowvar=False, bias=True)

log10mass = np.mean(data['x'][1], axis=0)
var_log10mass = np.std(data['x'][1], axis=0)**2

analysis_list = []

analysis_count = dict()
analysis_count['data_cov_count'] = cov_count 
analysis_count['data_var_log10mass'] = var_log10mass
analysis_count['data_count'] = count 
analysis_count['data_log10mass'] = log10mass
analysis_count['Gamma'] = 0.7
analysis_count['summary_stat'] = 'count_only'
analysis_count['which_cov_count'] = 'analytical'
analysis_count['which_cov_log10mass'] = 'analytical'
analysis_count['SSC_count_covariance'] = True
analysis_count['name_save'] = 'chains_count_only'
analysis_count['use_fiducial_data_vector'] = False

analysis_mass = copy.deepcopy(analysis_count)
analysis_mass['summary_stat'] = 'mass_only'
analysis_mass['name_save'] = 'chains_mass_only'

analysis_count_mass = copy.deepcopy(analysis_count)
analysis_count_mass['summary_stat'] = 'count_mass'
analysis_count_mass['name_save'] = 'chains_count_mass'

# analysis = dict()
# analysis['filename'] = '../euclid_flagship_simulations/flagship_cluster_catalogue_summary_statstics_DES_MoR_no_log10mass_obs_scatter.npy'
# analysis['Gamma'] = 0.7
# analysis['summary_stat'] = 'count_only'
# analysis['SSC_count_covariance'] = True
# analysis['name_save'] = 'chains_count_only'
# analysis['use_fiducial_data_vector'] = True

# analysis1 = copy.deepcopy(analysis)
# analysis1['summary_stat'] = 'mass_only'
# analysis1['name_save'] = 'chains_mass_only'

# analysis2 = copy.deepcopy(analysis)
# analysis2['summary_stat'] = 'count_mass'
# analysis2['name_save'] = 'chains_count_mass'

# #########

# analysis3 = copy.deepcopy(analysis)
# analysis3['use_fiducial_data_vector'] = False
# analysis3['name_save'] = 'chains_count_only_flagship'

# analysis4 = copy.deepcopy(analysis1)
# analysis4['use_fiducial_data_vector'] = False
# analysis4['name_save'] = 'chains_mass_only_flagship'

# analysis5 = copy.deepcopy(analysis2)
# analysis5['use_fiducial_data_vector'] = False
# analysis5['name_save'] = 'chains_count_mass_flagship'

analysis_list = [analysis_count, analysis_mass, analysis_count_mass]