import numpy as np
import sys
import glob
import pickle
import pandas as pd
import pyccl as ccl
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import pyccl as ccl
import clmm
from scipy import stats
from clmm import Cosmology

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()

sys.path.append('../modules/')
import model_completeness as comp
import model_purity as pur
import model_halo_mass_function as hmf
import model_cluster_abundance as cl_count
import model_stacked_cluster_mass as cl_mass
import pinocchio_binning_scheme as binning_scheme

#location@CC-IN2P3
where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'
file=glob.glob(where_cat)

def pinocchio_sim(index_simu=1):
    file_sim=file[index_simu]
    dat = pd.read_csv(file_sim ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
    ra, dec, redshift, Mvir_true = dat['ra'], dat['dec'], dat['z'], dat['M']/0.6777
    mask = np.log10(Mvir_true) > 14.2
    ra, dec, redshift, Mvir_true = ra[mask], dec[mask], redshift[mask], Mvir_true[mask]
    return redshift, Mvir_true

logm_edges = binning_scheme.logm_edges
redshift_edges = binning_scheme.redshift_edges
Z_bin = binning_scheme.Z_bin
LogMass_bin = binning_scheme.LogMass_bin

data = {}
data['redshift_bins'] =  Z_bin
data['log10Mass_bin'] = LogMass_bin
data['mean_log10mass_mass_redshift_per_sim'] = []
data['count_mass_redshift_per_sim'] = []

n_simu = 1000
for index_simu in range(len(file[:n_simu])):
    
    redshift, Mvir = pinocchio_sim(index_simu=index_simu)
    
    #summary_statistics
    N_mass_redshift, a, b = np.histogram2d(np.log10(Mvir), redshift, bins = [logm_edges, redshift_edges, ])
    Mean_log10mass_mass_redshift = stats.binned_statistic_2d(np.log10(Mvir), redshift, np.log10(Mvir), 'mean', bins=[logm_edges, redshift_edges]).statistic
    
    data['mean_log10mass_mass_redshift_per_sim'].append(Mean_log10mass_mass_redshift)
    data['count_mass_redshift_per_sim'].append(N_mass_redshift)
    
    if index_simu >= n_simu: break

Mean_mean_log10mass_mass_redshift = np.mean(data['mean_log10mass_mass_redshift_per_sim'], axis=0)
std_Mean_mean_log10mass_mass_redshift = np.std(data['mean_log10mass_mass_redshift_per_sim'], axis=0)

Mean_mean_count_mass_redshift = np.mean(data['count_mass_redshift_per_sim'], axis=0)
std_Mean_mean_count_mass_redshift = np.std(data['count_mass_redshift_per_sim'], axis=0)
count_ordered = np.zeros([n_simu, len(LogMass_bin)*len(Z_bin)])

for i in range(n_simu):
    count_ordered[i,:]=data['count_mass_redshift_per_sim'][i].flatten()
Covariance_count_estimation = np.cov(count_ordered.T, bias=True)

mass_ordered = np.zeros([n_simu, len(LogMass_bin)*len(Z_bin)])
for i in range(n_simu):
    mass_ordered[i,:]=data['mean_log10mass_mass_redshift_per_sim'][i].flatten()
Covariance_mass_estimation = np.cov(mass_ordered.T, bias=True)

data['mean_log10mass_mass_redshift'] = Mean_mean_log10mass_mass_redshift
data['err_mean_log10mass_mass_redshift'] = std_Mean_mean_log10mass_mass_redshift 
data['Cov_mean_log10mass_mass_redshift'] = Covariance_mass_estimation

data['mean_count_mass_redshift'] = Mean_mean_count_mass_redshift
data['err_mean_count_mass_redshift'] = std_Mean_mean_count_mass_redshift 
data['Cov_count_mass_redshift'] = Covariance_count_estimation

data.pop('mean_log10mass_mass_redshift_per_sim', None)
data.pop('count_mass_redshift_per_sim', None)

save_pickle(data, f'./data/pinocchio_data_vector/data_vector_pinocchio_mock_mass-redshift_bins.pkl', )
