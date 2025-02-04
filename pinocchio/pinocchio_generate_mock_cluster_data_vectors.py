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

import pinocchio_mass_richness_relation as sim_mr_rel
import pinocchio_binning_scheme as binning_scheme

sys.path.append('../modules/')
import class_richness_mass_relation as rm_relation

where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'

log10m0, z0 = sim_mr_rel.log10m0, sim_mr_rel.z0
proxy_mu0, proxy_muz, proxy_mulog10m =  sim_mr_rel.proxy_mu0, sim_mr_rel.proxy_muz, sim_mr_rel.proxy_mulog10m
proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m =  sim_mr_rel.proxy_sigma0, sim_mr_rel.proxy_sigmaz, sim_mr_rel.proxy_sigmalog10m
theta_rm = [log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m]
which_model = sim_mr_rel.which_model
sigma_wl_log10mass = sim_mr_rel.sigma_wl_log10mass
sigma_wl_obs_log10mass = sim_mr_rel.sigma_wl_obs_log10mass
sigma_wl_tot_log10mass = (sigma_wl_log10mass**2 + sigma_wl_obs_log10mass**2)**(1/2)
RM = rm_relation.Richness_mass_relation()
RM.select(which = which_model)

file=glob.glob(where_cat)

def pinocchio_sim(index_simu=1):
    #generate mass, richness, redshift catalog
    file_sim=file[index_simu]
    dat = pd.read_csv(file_sim ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
    ra, dec, redshift, Mvir_true = dat['ra'], dat['dec'], dat['z'], dat['M']/0.6777
    mask = np.log10(Mvir_true) > 14.2
    ra, dec, redshift, Mvir_true = ra[mask], dec[mask], redshift[mask], Mvir_true[mask]
    log10Mvir_obs = np.log10(Mvir_true) + sigma_wl_tot_log10mass * np.random.randn(len(np.log10(Mvir_true)))
    Mvir = 10**log10Mvir_obs
    richness = np.exp(RM.lnLambda_random(np.log10(Mvir_true), redshift, theta_rm))
    return redshift, Mvir_true, Mvir, richness
    
logm_edges = binning_scheme.logm_edges
redshift_edges = binning_scheme.redshift_edges
richness_edges = binning_scheme.richness_edges
Z_bin = binning_scheme.Z_bin
LogMass_bin = binning_scheme.LogMass_bin
Richness_bin = binning_scheme.Richness_bin

data = {}
data['richness_bins'] = Richness_bin
data['redshift_bins'] =  Z_bin
data['log10Mass_bin'] = LogMass_bin
data['mean_mass_richness_redshift_per_sim'] = []
data['mean_mass_power_richness_redshift_per_sim'] = []
data['count_richness_redshift_per_sim'] = []

n_simu = 1000
for index_simu in range(len(file[:n_simu])):
    redshift, Mvir_true, Mvir, richness = pinocchio_sim(index_simu=index_simu)
    
    #summary_statistics per sim
    #count
    N_richness_redshift, a, b = np.histogram2d(richness, redshift, bins = [richness_edges, redshift_edges, ])
    #mean mass
    Mean_mass_richness_redshift = stats.binned_statistic_2d(richness, redshift, Mvir, 'mean', 
                                                                 bins=[richness_edges, redshift_edges]).statistic
    #mean mass **(1/3) M_ij = (<mass**(1/3)>_{ij})**3
    Mean_mass_power_richness_redshift = stats.binned_statistic_2d(richness, redshift, Mvir**(1/3), 'mean', 
                                                                 bins=[richness_edges, redshift_edges]).statistic**3

    #store summary statistics
    data['mean_mass_richness_redshift_per_sim'].append(Mean_mass_richness_redshift)
    data['mean_mass_power_richness_redshift_per_sim'].append(Mean_mass_power_richness_redshift)
    data['count_richness_redshift_per_sim'].append(N_richness_redshift)
    
    if index_simu >= n_simu: break

#mean mass
data['mean_mass_richness_redshift'] = np.mean(data['mean_mass_richness_redshift_per_sim'], axis=0)
data['err_mean_mass_richness_redshift'] = np.std(data['mean_mass_richness_redshift_per_sim'], axis=0)

#mean over sims 
data['mean_mass_power_richness_redshift'] = np.mean(np.array(data['mean_mass_power_richness_redshift_per_sim']), axis=0)
data['err_mean_mass_power_richness_redshift'] = np.std(np.array(data['mean_mass_power_richness_redshift_per_sim']), axis=0)

#mean count
data['mean_count_richness_redshift'] = np.mean(data['count_richness_redshift_per_sim'], axis=0)
data['err_mean_count_richness_redshift'] = np.std(data['count_richness_redshift_per_sim'], axis=0)

#count_ordered = np.zeros([n_simu, len(Richness_bin)*len(Z_bin)])
# for i in range(n_simu):
#     count_ordered[i,:]=data['count_richness_redshift_per_sim'][i].flatten()
# Covariance_count_estimation = np.cov(count_ordered.T, bias=True)
# mass_ordered = np.zeros([n_simu, len(Richness_bin)*len(Z_bin)])
# mass_ordered_power = np.zeros([n_simu, len(Richness_bin)*len(Z_bin)])
# for i in range(n_simu):
#     mass_ordered[i,:]=data['mean_mass_richness_redshift_per_sim'][i].flatten()
#     mass_ordered_power[i,:]=data['mean_mass_power_1_3_richness_redshift_per_sim'][i].flatten()**3
# Covariance_mass_estimation = np.cov(mass_ordered.T, bias=True)
# Covariance_mass_power_estimation = np.cov(mass_ordered_power.T, bias=True)

#data.pop('mean_mass_richness_redshift_per_sim', None)
#data.pop('mean_mass_power_1_3_richness_redshift_per_sim', None)
#data.pop('count_richness_redshift_per_sim', None)

sigma_wl_logmass = np.log(10) * sigma_wl_log10mass

save_pickle(data, f'./data/pinocchio_data_vector/data_vector_pinocchio_v2_mock_{which_model}_sigma_lnMwl={sigma_wl_logmass:.2f}.pkl', )