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

sys.path.append('/pbs/throng/lsst/users/cpayerne/capish/modules/')
import model_completeness as comp
import model_purity as pur
import model_halo_mass_function as hmf
import class_richness_mass_relation as rm_relation
import model_cluster_abundance as cl_count
import model_stacked_cluster_mass as cl_mass
import pinocchio_mass_richness_relation as sim_mr_rel
import pinocchio_binning_scheme as binning_scheme


where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'

log10m0, z0 = sim_mr_rel.log10m0, sim_mr_rel.z0
proxy_mu0, proxy_muz, proxy_mulog10m =  sim_mr_rel.proxy_mu0, sim_mr_rel.proxy_muz, sim_mr_rel.proxy_mulog10m
proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m =  sim_mr_rel.proxy_sigma0, sim_mr_rel.proxy_sigmaz, sim_mr_rel.proxy_sigmalog10m
theta_rm = [log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m]
which_model = sim_mr_rel.which_model
sigma_wl_log10mass = sim_mr_rel.sigma_wl_log10mass
RM = rm_relation.Richness_mass_relation()
RM.select(which = which_model)

where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'
file=glob.glob(where_cat)

def pinocchio_sim(index_simu=1):
    file_sim=file[index_simu]
    dat = pd.read_csv(file_sim ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
    ra, dec, redshift, Mvir_true = dat['ra'], dat['dec'], dat['z'], dat['M']/0.6777
    mask = np.log10(Mvir_true) > 14.2
    ra, dec, redshift, Mvir_true = ra[mask], dec[mask], redshift[mask], Mvir_true[mask]
    log10Mvir_obs = np.log10(Mvir_true) + sigma_wl_log10mass * np.random.randn(len(np.log10(Mvir_true)))
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
data['mean_log10mass_richness_redshift_per_sim'] = []
data['count_richness_redshift_per_sim'] = []

n_simu = 1000
for index_simu in range(len(file[:n_simu])):
    
    redshift, Mvir_true, Mvir, richness = pinocchio_sim(index_simu=index_simu)
    
    #summary_statistics
    N_richness_redshift, a, b = np.histogram2d(richness, redshift, bins = [richness_edges, redshift_edges, ])
    Mean_log10mass_richness_redshift = stats.binned_statistic_2d(richness, redshift, np.log10(Mvir), 'mean', bins=[richness_edges, redshift_edges]).statistic
    
    data['mean_log10mass_richness_redshift_per_sim'].append(Mean_log10mass_richness_redshift)
    data['count_richness_redshift_per_sim'].append(N_richness_redshift)
    
    if index_simu >= n_simu: break
    
Mean_mean_log10mass_richness_redshift = np.mean(data['mean_log10mass_richness_redshift_per_sim'], axis=0)
std_Mean_mean_log10mass_richness_redshift = np.std(data['mean_log10mass_richness_redshift_per_sim'], axis=0)

Mean_mean_count_richness_redshift = np.mean(data['count_richness_redshift_per_sim'], axis=0)
std_Mean_mean_count_richness_redshift = np.std(data['count_richness_redshift_per_sim'], axis=0)
count_ordered = np.zeros([n_simu, len(Richness_bin)*len(Z_bin)])

for i in range(n_simu):
    count_ordered[i,:]=data['count_richness_redshift_per_sim'][i].flatten()
Covariance_count_estimation = np.cov(count_ordered.T, bias=True)

mass_ordered = np.zeros([n_simu, len(Richness_bin)*len(Z_bin)])
for i in range(n_simu):
    mass_ordered[i,:]=data['mean_log10mass_richness_redshift_per_sim'][i].flatten()
Covariance_mass_estimation = np.cov(mass_ordered.T, bias=True)

data['mean_log10mass_richness_redshift'] = Mean_mean_log10mass_richness_redshift
data['err_mean_log10mass_richness_redshift'] = std_Mean_mean_log10mass_richness_redshift 
data['Cov_mean_log10mass_richness_redshift'] = Covariance_mass_estimation

data['mean_count_richness_redshift'] = Mean_mean_count_richness_redshift
data['err_mean_count_richness_redshift'] = std_Mean_mean_count_richness_redshift 
data['Cov_count_richness_redshift'] = Covariance_count_estimation

data.pop('mean_log10mass_richness_redshift_per_sim', None)
data.pop('count_richness_redshift_per_sim', None)

sigma_wl_logmass = np.log(10) * sigma_wl_log10mass

save_pickle(data, f'/pbs/throng/lsst/users/cpayerne/capish/pinocchio_data_vector/data_vector_pinocchio_mock_{which_model}_sigma_lnMwl={sigma_wl_logmass:.2f}.pkl', )