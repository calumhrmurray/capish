import sys
import pyccl as ccl
import numpy as np
from multiprocessing import Pool
import emcee
import matplotlib.pyplot as plt
import time
import pickle
import logging
logger = logging.getLogger('mcmc')
logging.basicConfig(
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M:%S",
     level=logging.INFO,
 )

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
def load(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

sys.path.append('/pbs/throng/lsst/users/cpayerne/capish/modules/')
import cluster_abundance_covariance as cl_covar 
import model_cluster_abundance as cl_count
import class_likelihood as likelihood
CLCount_likelihood = likelihood.Likelihood()


code, fit_cosmo_str = sys.argv
if fit_cosmo_str == 'free_cosmo': fit_cosmo = True 
if fit_cosmo_str == 'fix_cosmo': fit_cosmo = False 

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

log10m0, z0 = np.log10(10**14.3), .5
proxy_mu0, proxy_muz, proxy_mulog10m =  3.2,0.078,2.22
proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m =  0.56,0,0.1
theta_rm = [log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m]

#data
index_sim = 0
data = load(F'/pbs/throng/lsst/users/cpayerne/capish/pinocchio_data_vector/data_vector_pinocchio_mock_{index_sim}.pkl')
Z_bin = data['redshift_bins']
Richness_bin = data['richness_bins']
N_obs = data['count_richness_redshift']

richness_grid = np.logspace(np.log10(np.min(Richness_bin)), np.log10(np.max(Richness_bin)), 310)
logm_grid = np.linspace(14.2, 16, 100)
z_grid = np.linspace(np.min(Z_bin), np.max(Z_bin), 151)

bins = {'redshift_bins':Z_bin, 'richness_bins': Richness_bin}
grids = {'logm_grid': logm_grid, 'z_grid': z_grid, 'richness_grid':richness_grid}
count_modelling = {'dNdzdlogMdOmega':None,'richness_mass_relation':None, 
                   'completeness':None, 'purity':None }
params = {'params_richness_mass_relation': theta_rm,
          'model_richness_mass_relation': 'log_normal',
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
    
    if fit_cosmo==True:
        proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m, Omegam, Sigma8 = theta
        if Omegam < 0: return -np.inf
        if Omegam > 1: return -np.inf
        if Sigma8 < 0: return -np.inf
        if Sigma8 > 1: return -np.inf
    
    else: proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta
    #mean parameter priors
    if proxy_mu0 < 0: return -np.inf
    #
    if proxy_muz < -2: return -np.inf
    if proxy_muz > 2: return -np.inf
    #
    if proxy_mulog10m < 0: return -np.inf
    #dispersion parameter priors
    if proxy_sigma0 < 0: return -np.inf
    #
    if proxy_sigmaz < -2: return -np.inf
    if proxy_sigmaz > 2: return -np.inf
    #
    if proxy_sigmalog10m < -2: return -np.inf
    if proxy_sigmalog10m > 2: return -np.inf

def lnL(theta, fit_cosmo):


    if fit_cosmo:
        proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m, Omegam, Sigma8 = theta
    else: proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta

    prior(theta, fit_cosmo)

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
    print(N)
    print(N_obs)
    gaussian=True
    if gaussian:
        NAverageHaloBias = Omega * cl_count.Cluster_NHaloBias_ProxyZ(bins, integrand_count = integrand_count_new,
                                                                     halo_bias = count_modelling_new['halo_bias'], 
                                                                     grids = grids, cosmo = cosmo)
        CLCovar = cl_covar.Covariance_matrix()
        NNSbb = CLCovar.sample_covariance_full_sky(Z_bin, Richness_bin, NAverageHaloBias.T, Sij_partialsky)
        Cov_tot = NNSbb + np.diag(N.T.flatten())
        CLCount_likelihood.lnLikelihood_Binned_Gaussian(N, N_obs, Cov_tot)
        lnLCLCount = CLCount_likelihood.lnL_Binned_Gaussian

    else:
        CLCount_likelihood.lnLikelihood_Binned_Poissonian(N, N_obs.T)
        lnLCLCount = CLCount_likelihood.lnL_Binned_Poissonian

    return lnLCLCount

if fit_cosmo:
    initial = [proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m, Omegam_true, sigma8_true]
    labels = [r'\ln \lambda_0', r'\mu_z', r'\mu_m', r'\sigma_{\ln \lambda, 0}', r'\sigma_z', r'\sigma_m', r'\Omega_m', r'\sigma_8']
else:
    initial = [proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m,]
    labels = [r'\ln \lambda_0', r'\mu_z', r'\mu_m', r'\sigma_{\ln \lambda, 0}', r'\sigma_z', r'\sigma_m', ]

ndim=len(initial)
t = time.time()
logger.info('Test likelihood')
logger.info(lnL(initial, fit_cosmo))
tf = time.time()
logger.info('time:' +str(tf-t))
nwalker = 100
pos = np.array(initial) + .01*np.random.randn(nwalker, ndim)
sampler = emcee.EnsembleSampler(nwalker, ndim, lnL, args=[fit_cosmo])
sampler.run_mcmc(pos, 200, progress=True);
flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
results={'flat_chains':flat_samples,'label_parameters':labels}
save_pickle(results, f'/pbs/throng/lsst/users/cpayerne/capish/chains/pinochio_chain_{index_sim}_{fit_cosmo_str}.pkl')
