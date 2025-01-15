import sys
import pyccl as ccl
import numpy as np
from multiprocessing import Pool
import emcee
import matplotlib.pyplot as plt
import time
import pickle
import logging
import argparse

sys.path.append('/pbs/throng/lsst/users/cpayerne/capish/modules/')
import model_completeness as comp
import model_purity as pur
import model_halo_mass_function as hmf
import class_richness_mass_relation as rm_relation
import model_cluster_abundance as cl_count
import model_stacked_cluster_mass as cl_mass
import pinocchio_mass_richness_relation as sim_mr_rel
import class_likelihood as likelihood
import pinocchio_binning_scheme as binning_scheme
import cluster_abundance_covariance as cl_covar

def collect_argparser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--type", type=str, required=False, default='NxMwl')
    parser.add_argument("--fit_cosmo", type=str, required=False, default='False')
    parser.add_argument("--number_params_scaling_relation", type=int, required=False, default=4)
    #parser.add_argument("--index_sim", type=int, required=False, default=0)
    return parser.parse_args()

analysis = collect_argparser()
type_analysis = analysis.type
logger = logging.getLogger('mcmc')
logging.basicConfig(
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M:%S",
     level=logging.INFO,)

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
def load(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

CLCount_likelihood = likelihood.Likelihood()

number_params_scaling_relation = analysis.number_params_scaling_relation
fit_cosmo = True if analysis.fit_cosmo == 'True' else False

# cosmology
Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711
True_value = [Omega_c_true + Omega_b_true, sigma8_true]

cosmo = ccl.Cosmology(Omega_c = Omegam_true - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)
#halo model
massdef = ccl.halos.massdef.MassDef('vir', 'critical',)
hmd = ccl.halos.hmfunc.MassFuncDespali16(mass_def=massdef)

log10m0, z0 = sim_mr_rel.log10m0, sim_mr_rel.z0
proxy_mu0, proxy_muz, proxy_mulog10m =  sim_mr_rel.proxy_mu0, sim_mr_rel.proxy_muz, sim_mr_rel.proxy_mulog10m
proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m =  sim_mr_rel.proxy_sigma0, sim_mr_rel.proxy_sigmaz, sim_mr_rel.proxy_sigmalog10m
which_model = sim_mr_rel.which_model
sigma_wl_log10mass = sim_mr_rel.sigma_wl_log10mass
theta_rm = [log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m]

#data
where = '/pbs/throng/lsst/users/cpayerne/capish/data/pinocchio_data_vector/'
data = load(where+f'data_vector_pinocchio_mock_{which_model}_sigma_lnMwl={sigma_wl_log10mass*np.log(10):.2f}.pkl')
redshift_edges = binning_scheme.redshift_edges
richness_edges = binning_scheme.richness_edges
Z_bin = binning_scheme.Z_bin
Richness_bin = binning_scheme.Richness_bin

N_obs = data['mean_count_richness_redshift']
log10Mwl_obs = data['mean_log10mass_richness_redshift']
err_log10Mwl_obs = data['err_mean_log10mass_richness_redshift']

richness_grid = np.logspace(np.log10(np.min(Richness_bin)), np.log10(np.max(Richness_bin)), 310)
logm_grid = np.linspace(14.2, 16, 100)
z_grid = np.linspace(np.min(Z_bin), np.max(Z_bin), 151)

bins = {'redshift_bins':Z_bin, 'richness_bins': Richness_bin}
grids = {'logm_grid': logm_grid, 'z_grid': z_grid, 'richness_grid':richness_grid}
count_modelling = {'dNdzdlogMdOmega':None,'richness_mass_relation':None, 
                   'completeness':None, 'purity':None }
params = {'params_richness_mass_relation': theta_rm,
          'model_richness_mass_relation': which_model,
          'CCL_cosmology': cosmo, 
          'halo_mass_distribution': hmd, 
          'params_concentration_mass_relation':'Duffy08', }

adds = {'add_purity' : False, 'add_completeness':False}
compute = {'compute_dNdzdlogMdOmega':True,
           'compute_richness_mass_relation':True, 
           'compute_completeness':False, 
           'compute_purity':False ,
           'compute_halo_bias':True,
           'compute_dNdzdlogMdOmega_log_slope': False}

logger.info('[load theory]: compute HMF+bias mass-redshift grids at fixed cosmology')

count_modelling_new = cl_count.recompute_count_modelling(count_modelling, grids = grids, compute = compute, params = params)

logger.info('[load theory]: Compute Sij matrix (SSC) from PySSC (Lacasa et al.)')

f_sky = 0.25
CLCovar = cl_covar.Covariance_matrix()
Sij_partialsky = CLCovar.compute_theoretical_Sij(Z_bin, cosmo, f_sky)

def prior(theta, fit_cosmo):
    
    if number_params_scaling_relation == 4:
        if fit_cosmo:
            proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, Omegam, Sigma8 = theta
        else: proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0 = theta
    
    elif number_params_scaling_relation == 6:
        if fit_cosmo:
            proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m, Omegam, Sigma8 = theta
        else: proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta
    
    if proxy_mu0 < 0: return 1
    if proxy_muz < -2: return 1
    if proxy_muz > 2: return 1
    if proxy_mulog10m < 0: return 1

    if fit_cosmo:
        if Omegam < 0: return 1
        if Omegam > 0.6: return 1
        if Sigma8 < 0.4: return 1
        if Sigma8 > 1: return 1
    #mean parameter priors
    
    if number_params_scaling_relation == 6:
        if proxy_sigmaz < -2: return 1
        if proxy_sigmaz > 2: return 1
        if proxy_sigmalog10m < -2: return 1
        if proxy_sigmalog10m > 2: return 1
    
    return 0

def lnL(theta, fit_cosmo, likelihood_used):


    if number_params_scaling_relation == 4:
        if fit_cosmo==True:
            proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, Omegam, Sigma8 = theta
        else: 
            proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0 = theta
        
    if number_params_scaling_relation == 6:
        if fit_cosmo==True:
            proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m, Omegam, Sigma8 = theta
        else: 
            proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta

    prior_value = prior(theta, fit_cosmo)
    if prior_value == 1: return -np.inf

    if number_params_scaling_relation == 4:
        theta_rm_new = [log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, 0, 0]
    if number_params_scaling_relation == 6:
        theta_rm_new = [log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m]

    if fit_cosmo:
        cosmo_new = ccl.Cosmology(Omega_c = Omegam - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = Sigma8, n_s=0.96)
    else: cosmo_new = cosmo

    params_new = {'params_richness_mass_relation': theta_rm_new,
                  'model_richness_mass_relation': params['model_richness_mass_relation'],
                  'CCL_cosmology': cosmo_new, 
                  'halo_mass_distribution': hmd}

    compute_new = {'compute_dNdzdlogMdOmega':fit_cosmo,
                   'compute_richness_mass_relation':True, 
                   'compute_completeness':False, 
                   'compute_purity':False,
                   'compute_halo_bias':False,
                  'compute_dNdzdlogMdOmega_log_slope': False}

    adds_N = {'add_purity':False, 
              'add_completeness':False}

    count_modelling_new = cl_count.recompute_count_modelling(count_modelling, grids = grids, compute = compute_new, params = params_new)
    test_sign = count_modelling_new['richness_mass_relation - sigma'].flatten() < 0
    if len(test_sign[test_sign==True]) != 0: return -np.inf
    integrand_count_new = cl_count.define_count_integrand(count_modelling_new, adds_N)
    Omega = 4*np.pi*f_sky

    N = Omega * cl_count.Cluster_SurfaceDensity_ProxyZ(bins, integrand_count = integrand_count_new, grids = grids)
    
    lnL = 0
    if 'N' in analysis.type:
        if likelihood_used=='Gaussian+SSC':
            NAverageHaloBias = Omega * cl_count.Cluster_NHaloBias_ProxyZ(bins, integrand_count = integrand_count_new,
                                                                         halo_bias = count_modelling_new['halo_bias'], 
                                                                         grids = grids, cosmo = cosmo)
            CLCovar = cl_covar.Covariance_matrix()
            NNSbb = CLCovar.sample_covariance_full_sky(Z_bin, Richness_bin, NAverageHaloBias.T, Sij_partialsky)
            Cov_tot = NNSbb + np.diag(N.T.flatten())
            CLCount_likelihood.lnLikelihood_Binned_Gaussian(N, N_obs, Cov_tot)
            lnLCLCount = CLCount_likelihood.lnL_Binned_Gaussian

        elif likelihood_used=='Poisson':
            CLCount_likelihood.lnLikelihood_Binned_Poissonian(N, N_obs.T)
            lnLCLCount = CLCount_likelihood.lnL_Binned_Poissonian
        lnL += lnLCLCount
        
    if 'Mwl' in analysis.type:
        Nlog10Mth = Omega * cl_mass.Cluster_dNd0mega_Mass_ProxyZ(bins, integrand_count = integrand_count_new, grids = grids, Nlog10m = True)
        log10Mth = Nlog10Mth/N
        lnLCLM = -.5*np.sum(((log10Mth - log10Mwl_obs)/err_log10Mwl_obs)**2)
        lnL += lnLCLM
    return lnL

initial =  [proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0]
initial += [proxy_sigmaz, proxy_sigmalog10m] if number_params_scaling_relation == 6 else []
initial += [Omegam_true, sigma8_true] if fit_cosmo else []

labels =  [r'\ln \lambda_0', r'\mu_z', r'\mu_m', r'\sigma_{\ln \lambda, 0}']
labels += [r'\sigma_z', r'\sigma_m'] if number_params_scaling_relation == 6 else []
labels += [r'\Omega_m', r'\sigma_8'] if fit_cosmo else []

likelihood_used = 'Poisson'
fit_cosmo_str = 'fit_cosmo' if fit_cosmo else 'fixed_cosmo'
where_to_save = '/pbs/throng/lsst/users/cpayerne/capish/chains/'
name_save=where_to_save+f'pinochio_chain_{type_analysis}_{fit_cosmo_str}_num_params_rm_rel_{number_params_scaling_relation}_{which_model}_sigma_lnMwl={sigma_wl_log10mass*np.log(10):.2f}_with_{likelihood_used}_likelihood.pkl'
logger.info('[Saving chains]: The chains will be saved in the file '+ name_save)
ndim=len(initial)
t = time.time()
initial_str = ''
for i in range(len(initial)):
    initial_str = labels[i] + ' = '+str(initial[i])
    logger.info('[MCMC]: Initial position: ' + initial_str )

t = time.time()
lnL_start = lnL(initial, fit_cosmo)
tf = time.time()

logger.info(f'[First test]: lnL(initial) = {lnL_start:.2f} computed in {tf-t:.2f} seconds')

nwalker = 60
pos = np.array(initial) + .01*np.random.randn(nwalker, ndim)
sampler = emcee.EnsembleSampler(nwalker, ndim, lnL, args=[fit_cosmo, likelihood_used])
sampler.run_mcmc(pos, 200, progress=True);
flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
results={'flat_chains':flat_samples,'label_parameters':labels}
save_pickle(results, name_save)
