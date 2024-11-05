from astropy.table import Table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/pbs/home/c/cmurray/cluster_likelihood/modules/')
import simulation
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import corner
import pickle
import scipy.stats as stats
from matplotlib.cm import get_cmap

# Path to your FITS file
file_path = '/sps/euclid/Users/cmurray/clusters_likelihood/redmapper_y1a1_public_v6.4_catalog.fits.gz'

# Load the FITS file into an Astropy Table
redmapper_catalogue = Table.read(file_path, format='fits')


redmapper_catalogue.remove_columns( ['P_CEN', 'RA_CEN', 'DEC_CEN', 'ID_CEN', 'PZBINS', 'PZ'] )
redmapper_catalogue = redmapper_catalogue.to_pandas()

rich_bins = np.logspace( np.log10( 20 ) , np.log10( 240 ) , 50 )
rich_cents = ( rich_bins[1:] + rich_bins[:-1] )/2.

des_lambda_bins = np.array( [ 20 , 30 , 45 , 60 , 240 ])
# des_lambda_bins = np.array( [ 30 , 45 , 60 , 240 ])
des_z_bins = np.array( [ 0.2 , 0.35 , 0.5 , 0.65 ])

i = 0
z_idx_0 = ( redmapper_catalogue['Z_LAMBDA'] < des_z_bins[i+1] ) & ( redmapper_catalogue['Z_LAMBDA'] > des_z_bins[i] )
i = 1
z_idx_1 = ( redmapper_catalogue['Z_LAMBDA'] < des_z_bins[i+1] ) & ( redmapper_catalogue['Z_LAMBDA'] > des_z_bins[i] )
i = 2
z_idx_2 = ( redmapper_catalogue['Z_LAMBDA'] < des_z_bins[i+1] ) & ( redmapper_catalogue['Z_LAMBDA'] > des_z_bins[i] )

n_clusters_0 , _  = np.histogram( redmapper_catalogue['LAMBDA'][ z_idx_0 ], bins = des_lambda_bins )
n_clusters_1 , _  = np.histogram( redmapper_catalogue['LAMBDA'][ z_idx_1 ], bins = des_lambda_bins )
n_clusters_2 , _  = np.histogram( redmapper_catalogue['LAMBDA'][ z_idx_2 ], bins = des_lambda_bins )


# Mean mass results, Table II DES Y1 cluster abundance results
mwl_mean_0 = np.array( [ 14.036 , 14.323 , 14.454 , 14.758 ] )
mwl_mean_1 = np.array( [ 14.007 , 14.291 , 14.488 , 14.744 ] )
mwl_mean_2 = np.array( [ 13.929 , 14.301 , 14.493 , 14.724 ] )

# mwl_mean_0 = np.array( [ 14.323 , 14.454 , 14.758 ] )
# mwl_mean_1 = np.array( [ 14.291 , 14.488 , 14.744 ] )
# mwl_mean_2 = np.array( [ 14.301 , 14.493 , 14.724 ] )


# Std for aMean mass results, Table II DES Y1 cluster abundance results
mwl_std_0 = np.array( [ 0.032 + 0.045 , 0.031 + 0.051 , 0.044 + 0.050 , 0.038 + 0.052 ] )
mwl_std_1 = np.array( [ 0.033 + 0.056 , 0.031 + 0.061 , 0.044 + 0.065 , 0.038 + 0.052 ] )
mwl_std_2 = np.array( [ 0.048 + 0.072 , 0.041 + 0.086 , 0.056 + 0.068 , 0.061 + 0.069 ] )

# mwl_std_0 = np.array( [ 0.031 + 0.051 , 0.044 + 0.050 , 0.038 + 0.052 ] )
# mwl_std_1 = np.array( [ 0.031 + 0.061 , 0.044 + 0.065 , 0.038 + 0.052 ] )
# mwl_std_2 = np.array( [ 0.041 + 0.086 , 0.056 + 0.068 , 0.061 + 0.069 ] )


mwl_std = np.array( [ mwl_std_0, mwl_std_1, mwl_std_2 ] ).T

stacked_simulator = simulation.Universe_simulation( 'stacked_counts' ,
                                                    variable_params=['omega_m', 
                                                                     'sigma_8', 
                                                                     'h',
                                                                     'alpha' , 
                                                                     'B' ,
                                                                     'log10Mmin',
                                                                     'beta'
                                                                     'sigma' ],
                                                    fixed_params={'w_0': -1, 'w_a': 0 } )
stacked_simulator.selection_richness = 0
stacked_simulator.dOmega = 1500/41253 * 4*np.pi
stacked_simulator.richness_bins = des_lambda_bins
stacked_simulator.redshift_bins = des_z_bins
stacked_simulator.sigma_mwl = 0.3
stacked_simulator.include_mwl_measurement_errors = True
# measure errors for weak lensing masses
stacked_simulator.mwl_std = mwl_std
stacked_simulator.correlation_mass_evolution = False
stacked_simulator.set_richness_mass_relation( 'halo model' )
# set the Aemulus correction to the halo mass function
stacked_simulator.s = 0.037
stacked_simulator.q = 1.008

# set the binning used to simulate a cluster catalogue
stacked_simulator.set_bins( z_bins = np.arange( 0.15 , 0.7 , 0.05 ) , log10m_bins =  np.arange( 12.8, 16 , 0.01))

# Assuming the arrays are already of the same shape and are 2D
n_clusters = np.array( [ n_clusters_0, n_clusters_1, n_clusters_2 ]).T
mwl_mean = np.array( [ mwl_mean_0, mwl_mean_1, mwl_mean_2 ] ).T

# bit messy at the moment
from torch.distributions import Distribution, Uniform, Normal

# Define individual priors with float32 tensors
dtype = torch.float32
prior_om = Uniform(torch.tensor([0.05], dtype=dtype), torch.tensor([1.0], dtype=dtype))
prior_s8 = Uniform(torch.tensor([0.5], dtype=dtype), torch.tensor([1.5], dtype=dtype))
prior_h = Normal(torch.tensor([0.7], dtype=dtype), torch.tensor([0.1], dtype=dtype))  # Normal prior on h
prior_alpha = Uniform(torch.tensor([0.4], dtype=dtype), torch.tensor([1.2], dtype=dtype))
prior_B = Uniform(torch.tensor([np.log10(10)], dtype=dtype), torch.tensor([np.log10(30)], dtype=dtype))
prior_log10Mmin = Uniform(torch.tensor([10.0], dtype=dtype), torch.tensor([14.0], dtype=dtype))
prior_beta = Uniform(torch.tensor([-5.0], dtype=dtype), torch.tensor([5.0], dtype=dtype))
prior_sigma = Uniform(torch.tensor([0.05], dtype=dtype), torch.tensor([0.5], dtype=dtype))

# Combine the priors into a list
priors = [prior_om, prior_s8, prior_h, prior_alpha, prior_B, prior_log10Mmin, prior_beta, prior_sigma]


# Process the combined prior to be used in inference
#prior = process_prior(priors)

# Run inference
des_posterior = infer(
    stacked_simulator.run_simulation,
    priors,
    method="SNPE",
    num_simulations=60000,
    num_workers=128
)

with open('/sps/euclid/Users/cmurray/clusters_likelihood/des_posterior_halo_model_redshift_evolution_with_h_normal_prior.pkl', "wb") as handle:
    pickle.dump( des_posterior, handle)


print('Posterior saved')
