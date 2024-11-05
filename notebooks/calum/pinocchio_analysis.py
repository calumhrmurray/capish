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
from torch.distributions import Distribution, Uniform, Normal

pinocchio_mock = np.load('/pbs/home/c/cmurray/cluster_likelihood/pinocchio_data_vector/data_vector_pinocchio_mock_0.pkl' , allow_pickle= True )

richness_cents = np.array( [ ( 20 + 29.35 )/2. , ( 29.355 + 43.1 )/2. , ( 43.1 + 63.25 )/2. , ( 63.25 + 92.8 )/2. , ( 92.8 + 136.25 )/2. , ( 137.25 + 200 )/2.])

plt.figure( figsize = ( 12, 4 ))
plt.subplot(121)

for i in range( 0 , len( pinocchio_mock['count_richness_redshift'].T ) ):
    plt.plot( richness_cents , 
              pinocchio_mock['count_richness_redshift'].T[i] , drawstyle = 'steps-mid')

plt.xscale('log')
plt.yscale('log')
plt.yticks( [ 100 , 200 , 500 , 1000 , 2000 ], labels = [ 100 , 200 , 500 , 1000 , 2000 ])
plt.xlim( 20 , 200 )
plt.ylim( 100 , 6000 )

plt.subplot(122)

for i in range( 0 , len( pinocchio_mock['count_richness_redshift'].T ) ):
    plt.plot( richness_cents , 
              np.log10( pinocchio_mock['mean_mass_richness_redshift'].T[i] ) , drawstyle = 'steps-mid')

plt.xscale('log')
plt.xlim( 20 , 200 )
plt.ylim( 13.7 , 15 )

richness_bins = np.array( [ 20 , 29.35 , 43.1 , 63.2 , 92.83 , 136.26 , 200 ])
redshift_bins = np.array( [ 0.2 , 0.36 , 0.52 , 0.68 , 0.84 , 1.0 ])


stacked_simulator_pl = simulation.Universe_simulation( 'stacked_counts' ,
                                                        variable_params=['omega_m', 
                                                                         'sigma_8', 
                                                                         'h',
                                                                         'alpha' , 
                                                                         'c' , 
                                                                         'beta',
                                                                         'sigma' ],
                                                        fixed_params={'w_0': -1, 'w_a': 0 } )
stacked_simulator_pl.selection_richness = 0
stacked_simulator_pl.dOmega = 0.25 * 4*np.pi
stacked_simulator_pl.richness_bins = richness_bins
stacked_simulator_pl.redshift_bins = redshift_bins
stacked_simulator_pl.sigma_mwl = 0.3
stacked_simulator_pl.include_mwl_measurement_errors = False
#stacked_simulator_pl.mwl_std = mwl_std
stacked_simulator_pl.correlation_mass_evolution = False
stacked_simulator_pl.set_richness_mass_relation( 'power law' )
stacked_simulator_pl.set_bins( z_bins = np.arange( 0.2 , 1 , 0.05 ) , 
                               log10m_bins=  np.arange( 14.2 , 16, 0.01))
stacked_simulator_pl.hmf = ccl.halos.MassFuncDespali16(mass_def='vir')


# Define individual priors with correct tensor shape
prior_om = Uniform(torch.tensor([0.05]), torch.tensor([1.0]))
prior_s8 = Uniform(torch.tensor([0.5]), torch.tensor([1.5]))
prior_h = Normal(torch.tensor([0.7]), torch.tensor([0.1]))  # Normal prior on h
prior_alpha = Uniform(torch.tensor([0.4]), torch.tensor([1.2]))
prior_c = Uniform(torch.tensor([1.0]), torch.tensor([5.0]))
prior_beta = Uniform(torch.tensor([-5.0]), torch.tensor([5.0]))
prior_sigma = Uniform(torch.tensor([0.05]), torch.tensor([0.5]))

# Combine the priors into a list for processing
priors = [ prior_om, prior_s8, prior_h, prior_alpha, prior_c , prior_beta , prior_sigma ]

# infer posteriors
pinocchio_posterior = infer( stacked_simulator_pl.run_simulation , 
                             priors, 
                             method = 'SNPE', 
                             num_simulations = 60000 , 
                             num_workers = 128 )

with open('/sps/euclid/Users/cmurray/clusters_likelihood/pinocchio_power_law.pkl', "wb") as handle:
    pickle.dump( des_posterior, handle)


print('Posterior saved')


