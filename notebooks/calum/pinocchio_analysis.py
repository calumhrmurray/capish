from astropy.table import Table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/pbs/home/c/cmurray/cluster_likelihood/modules/')
import simulation
import pinocchio_binning_scheme as binning_scheme

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import corner
import pickle
import scipy.stats as stats
from matplotlib.cm import get_cmap
from torch.distributions import Distribution, Uniform, Normal
import pyccl as ccl
from matplotlib.cm import get_cmap

import pickle

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
def load(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

pinocchio_mock = np.load('/pbs/home/c/cmurray/cluster_likelihood/data/pinocchio_data_vector/data_vector_pinocchio_mock_log_normal_poisson_scatter_sigma_lnMwl=0.25.pkl' , allow_pickle= True )

richness_cents = np.array( [ ( np.array( pinocchio_mock['richness_bins'] ).T[0] + np.array( pinocchio_mock['richness_bins'] ).T[1] )/2. ])[0]
redshift_cents = np.array( [ ( np.array( pinocchio_mock['redshift_bins'] ).T[0] + np.array( pinocchio_mock['redshift_bins'] ).T[1] )/2. ])[0]

richness_bins = list(np.array(pinocchio_mock['richness_bins']).T[0])
richness_bins.append( np.array( pinocchio_mock['richness_bins'] ).T[1][-1]  )

redshift_bins = list(np.array( pinocchio_mock['redshift_bins']).T[0])
redshift_bins.append( np.array( pinocchio_mock['redshift_bins'] ).T[1][-1]  )

def dOmega_func( z ):
    return 0.25 * 4 *np.pi

stacked_simulator_pl = simulation.Universe_simulation( 'stacked_counts' ,
                                                        variable_params=['omega_m', 
                                                                         'sigma_8', 
                                                                         'alpha' , 
                                                                         'c' , 
                                                                         'beta',
                                                                         'sigma' ],
                                                        fixed_params={'w_0': -1, 'w_a': 0 , 'h':0.6777} )


stacked_simulator_pl.selection_richness = 0
stacked_simulator_pl.dOmega = dOmega_func

stacked_simulator_pl.sigma_mwl = 0.25
stacked_simulator_pl.include_mwl_measurement_errors = False
stacked_simulator_pl.correlation_mass_evolution = False
stacked_simulator_pl.set_richness_mass_relation( 'constantins model' )
stacked_simulator_pl.set_bins( z_bins = np.arange( 0.2 , 1 , 0.001 ) , 
                               log10m_bins=  np.arange( 14.2 , 15.5 , 0.001))
stacked_simulator_pl.massdef = ccl.halos.massdef.MassDef('vir', 'critical',)
stacked_simulator_pl.halobias_fct = ccl.halos.hbias.tinker10.HaloBiasTinker10(mass_def=stacked_simulator_pl.massdef)
stacked_simulator_pl.f_sky = 0.25
stacked_simulator_pl.hmf = ccl.halos.hmfunc.MassFuncDespali16( mass_def= stacked_simulator_pl.massdef )
stacked_simulator_pl.Z_edges_hybrid = binning_scheme.redshift_edges
Z_bin_hybrid = [[stacked_simulator_pl.Z_edges_hybrid[i], stacked_simulator_pl.Z_edges_hybrid[i+1]] for i in range(len(stacked_simulator_pl.Z_edges_hybrid)-1)]
# these get changed later
stacked_simulator_pl.use_hybrid = False
stacked_simulator_pl.poisson_only = False

stacked_simulator_pl.richness_bins = richness_bins
stacked_simulator_pl.redshift_bins = redshift_bins

Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711

cosmo = ccl.Cosmology(Omega_c = Omegam_true - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)

def compute_Sij_matrix(cosmo, Z_bin_hybrid, f_sky = 1):
    import cluster_abundance_covariance as cl_covar
    CLCovar = cl_covar.Covariance_matrix()
    Sij_partialsky_exact_standard = CLCovar.compute_theoretical_Sij(Z_bin_hybrid, cosmo, 
                                                                f_sky,
                                                                S_ij_type='full_sky_rescaled_approx', 
                                                                path=None)
    return Sij_partialsky_exact_standard

def compute_sigmaij_matrix(cosmo, z_grid, f_sky = 1):
    import cluster_abundance_covariance as cl_covar
    z_grid_center = np.array([(z_grid[i] + z_grid[i+1])/2 for i in range(len(z_grid)-1)])
    CLCovar = cl_covar.Covariance_matrix()
    sigmaij_partialsky_exact_standard = CLCovar.compute_theoretical_sigmaij(z_grid_center, cosmo, f_sky)
    return sigmaij_partialsky_exact_standard


Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711

cosmo = ccl.Cosmology(Omega_c = Omegam_true - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)

have_PySSC = False
if have_PySSC:
    z_grid = np.linspace(0.2, 1, 1000)
    stacked_simulator_pl.sigmaij_SSC = compute_sigmaij_matrix(cosmo, z_grid)/stacked_simulator_pl.f_sky
    save_pickle(stacked_simulator_pl.sigmaij_SSC*stacked_simulator_pl.f_sky, f'/pbs/throng/lsst/users/cpayerne/capish/data/pinocchio_matrix/sigma_ij_full_sky.pkl', )
    
    stacked_simulator_pl.Sij_SSC = compute_Sij_matrix(cosmo, Z_bin_hybrid)/stacked_simulator_pl.f_sky
    save_pickle(stacked_simulator_pl.Sij_SSC*stacked_simulator_pl.f_sky, f'/pbs/throng/lsst/users/cpayerne/capish/data/pinocchio_matrix/S_ij_full_sky.pkl', )
else: 
    S_ij_full_sky = load(f'/pbs/throng/lsst/users/cpayerne/capish/data/pinocchio_matrix/S_ij_full_sky.pkl' )
    stacked_simulator_pl.Sij_SSC = S_ij_full_sky/stacked_simulator_pl.f_sky
    
    sigma_ij_full_sky = load(f'/pbs/throng/lsst/users/cpayerne/capish/data/pinocchio_matrix/sigma_ij_full_sky.pkl' )
    stacked_simulator_pl.sigmaij_SSC = sigma_ij_full_sky/stacked_simulator_pl.f_sky


# Define individual priors with correct tensor shape
prior_om = Uniform(torch.tensor([0.25]), torch.tensor([0.4]))
prior_s8 = Uniform(torch.tensor([0.8]), torch.tensor([1.]))
prior_h = Normal(torch.tensor([0.6777]), torch.tensor([0.001]))  # Normal prior on h
prior_alpha = Uniform(torch.tensor([1.8]), torch.tensor([2.6]))
prior_c = Uniform(torch.tensor([3.0]), torch.tensor([3.5]))
prior_beta = Uniform(torch.tensor([-0.3]), torch.tensor([0.3]))
prior_sigma = Uniform(torch.tensor([0.3]), torch.tensor([0.6]))

# Combine the priors into a list for processing
priors = [ prior_om, prior_s8, prior_h, prior_alpha, prior_c , prior_beta , prior_sigma ]

# decide if you want hybrid, SSC etc.
stacked_simulator_pl.use_hybrid = True
stacked_simulator_pl.poisson_only = True

# infer the posterior calculator
pinocchio_posterior_calculator = infer( stacked_simulator_pl.run_simulation , 
                             priors, 
                             method = 'SNPE', 
                             num_simulations = 30000 , 
                             num_workers = 20 )

# # save the posterior calculator
# with open('/sps/euclid/Users/cmurray/clusters_likelihood/pinocchio_posterior_calculator_SSC.pkl', "wb") as handle:
#     pickle.dump( pinocchio_posterior_calculator, handle)

# # save the posterior calculator
# with open('/sps/euclid/Users/cmurray/clusters_likelihood/pinocchio_posterior_calculator_poisson_only.pkl', "wb") as handle:
#     pickle.dump( pinocchio_posterior_calculator, handle)

# # save the posterior calculator
# with open('/sps/euclid/Users/cmurray/clusters_likelihood/pinocchio_posterior_calculator_SSC_hybrid.pkl', "wb") as handle:
#     pickle.dump( pinocchio_posterior_calculator, handle)

# save the posterior calculator
with open('/sps/euclid/Users/cmurray/clusters_likelihood/pinocchio_posterior_calculator_poisson_hybrid.pkl', "wb") as handle:
    pickle.dump( pinocchio_posterior_calculator, handle)



