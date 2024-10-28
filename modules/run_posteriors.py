import numpy as np
import matplotlib.pyplot as plt
import torch

from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import pickle
import pyccl as ccl

class Universe_simulation:
    
    def __init__( self, summary_statistic , for_simulate_for_sbi = False ):
        """
        Initialize the Simulation class.
        """
        if summary_statistic == 'stacked_counts':
            self.summary_statistic = self.stacked_counts
        elif summary_statistic == 'unbinned_counts':
            self.summary_statistic = self.unbinned_counts
        else:
            print('Your chosen summary statistic is unknown:' , summary_statistic )
        self.selection_richness = 20
        self.for_simulate_for_sbi = for_simulate_for_sbi
        # hmf properties
        self.dlog10m = 0.01
        self.log10ms = np.arange( 13.5 , 15.5 , self.dlog10m )
        self.Ms = 10**self.log10ms
        self.hmf = ccl.halos.MassFuncTinker10( mass_def='200m' )
        # Create a grid of mass and redshift values
        self.z_bins = np.arange( 0 , 1.2 , 0.05 )
        self.zs = ( self.z_bins[1:] + self.z_bins[:-1] )/2.
        mass_grid, redshift_grid = np.meshgrid( self.log10ms , self.zs )
        self.mass_grid = mass_grid
        self.redshift_grid = redshift_grid
        # Flatten the mass and redshift grids
        self.mass_values = self.mass_grid.flatten()
        self.redshift_values = self.redshift_grid.flatten()
        # for the stacked counts
        self.richness_bins = np.logspace( np.log10( 20 ), np.log10( 300 ), 15 )
        self.redshift_bins = np.linspace( 0.025 , 1.2 , 6 ) 
        # for the "unbinned" counts
        self.small_log10Mwl_bins = np.arange( 12.5 , 15.5 , 0.05 )
        self.small_richness_bins = np.logspace( np.log10( 20 ), np.log10( 300 ), 30 )
        self.small_redshift_bins = np.arange( 0 , 1.2 , 0.1 )
        # old bins
#         self.small_log10Mwl_bins = np.arange( 12.5 , 15.5 , 0.1 )
#         self.small_richness_bins = np.logspace( np.log10( 20 ), np.log10( 300 ), 15 )
#         self.small_redshift_bins = np.linspace( 0.025 , 1.125 , 6 ) 
        # survey size
        self.dOmega = 0.5 * 4*np.pi
        # fixed parameters
        self.alpha_mwl = 1
        self.sigma_mwl = 0.3 
        self.c_mwl = np.log( 1e14 )
        self.r = 0
        self.H0 = 70
        # transfer function
        #self.transfer_function = 'eisenstein_hu'
        self.transfer_function = 'boltzmann_camb'
        

    def run_simulation( self, parameter_set ):
        """
        Run the simulation with the given parameters.
        """
        richness, log10M_wl, z_clusters = self._run_simulation( parameter_set )
        
        if self.for_simulate_for_sbi:
            return torch.tensor( self.summary_statistic( richness, log10M_wl, z_clusters ) )
        else:
            return self.summary_statistic( richness, log10M_wl, z_clusters )
    

    def _run_simulation( self, parameter_set ):
        """
        Run the simulation. This contains the selection function!
        """
        
        # get the latent cluster properties, a poisson realisation
        mu_clusters, z_clusters = self.get_cluster_catalogue( parameter_set )
        
        # get the observed cluster properties
        richness, log10M_wl = self.mass_observable_relation(mu_clusters, z_clusters , parameter_set )
        
        # apply selection function
        selection = richness > self.selection_richness
        return richness[selection], log10M_wl[selection], z_clusters[selection]
        

    def get_cluster_catalogue(self, parameter_set):
        Om0, sigma8, alpha_l, sigma_l, c_l = parameter_set

        # Ensure that parameters are native Python floats (not PyTorch tensors)
        Om0 = float(Om0)
        sigma8 = float(sigma8)

        # Create the CCL Cosmology object once
        cosmo = ccl.Cosmology(Omega_c = Om0 - 0.05,
                              Omega_b = 0.05,
                              h = self.H0 / 100,
                              n_s = 0.96,
                              sigma8 = sigma8,
                              Omega_k = 0.0,
                              transfer_function=self.transfer_function,
                              matter_power_spectrum='linear')

        dz = 0.05
        z_bins = np.arange( 0, 1.2, dz )
        z_bin_centers = (z_bins[:-1] + z_bins[1:]) / 2.0
        scale_factor_bins = 1/( z_bins + 1 )
        scale_factors = 1 / (z_bin_centers + 1)

        # Compute comoving volumes only once
        dV = cosmo.comoving_volume_element( scale_factors ) #cosmo.comoving_volume( scale_factor_bins[1:]) - cosmo.comoving_volume( scale_factor_bins[:-1] )
        da = scale_factor_bins[:-1] - scale_factor_bins[1:]
        
        cluster_abundance = []

        for i, a in enumerate(scale_factors):
            # Calculate halo mass function for the current redshift (as scalar `a`)
            dndlog10M = self.hmf( cosmo, self.Ms, a )

            # Compute counts in each bin
            counts_per_bin = np.random.poisson( dndlog10M * dV[i] * self.dlog10m * self.dOmega * da[i] )
            cluster_abundance.append(counts_per_bin)

        cluster_abundance = np.array(cluster_abundance).flatten()

        # Use np.repeat to create the catalog based on counts in cluster_abundance
        cat_mass = np.repeat(self.mass_values, cluster_abundance)
        cat_redshift = np.repeat(self.redshift_values, cluster_abundance)
        cat_mu = np.log( 10 ** cat_mass / 1e14 )

        return cat_mu, cat_redshift

    def mass_observable_relation( self, mu , z , parameter_set ):
        
        Om0, sigma8 , alpha_l , sigma_l , c_l =  parameter_set
        alpha_l = alpha_l.numpy()
        sigma_l = sigma_l.numpy()
        c_l = c_l.numpy()

        mean_l = c_l + alpha_l * mu
        mean_mwl = self.c_mwl + self.alpha_mwl * mu

        cov = [ [ sigma_l**2 , self.r * self.sigma_mwl * sigma_l ] , 
                [ self.r * self.sigma_mwl * sigma_l , self.sigma_mwl**2 ] ]

        noise = np.random.multivariate_normal( [ 0 , 0 ] , cov = cov  , size = len( mu ) )

        ln_richness = mean_l + noise.T[0]
        lnM_wl = mean_mwl + noise.T[1]


        return np.exp( ln_richness  ) , np.log10( np.exp( lnM_wl ) )
    

    def unbinned_counts(self, richness, log10M_wl, z_clusters ):
        """
        Calculate the number of clusters in bins of cluster richness, redshift, and weak lensing mass.

        Parameters:
        - richness: array-like, cluster richness values
        - log10M_wl: array-like, log10 of weak lensing mass values
        - z_clusters: array-like, redshift values of the clusters
        - richness_bins: array-like, edges of the bins for richness
        - redshift_bins: array-like, edges of the bins for redshift
        - mass_bins: array-like, edges of the bins for weak lensing mass (log10)

        Returns:
        - counts: a 3D array of shape (len(richness_bins)-1, len(redshift_bins)-1, len(mass_bins)-1)
        """

        # Calculate histogram counts in 3D bins
        counts, edges = np.histogramdd(
            np.column_stack([ richness, 
                              z_clusters, 
                              log10M_wl ]),
            bins=[ self.small_richness_bins, 
                   self.small_redshift_bins, 
                   self.small_log10Mwl_bins ]
        )

        return counts.flatten()
    
    def stacked_counts( self , richness, log10M_wl, redshift ):
        """
        Calculate the number of clusters in bins of cluster richness and redshift,
        and calculate the mean cluster weak-lensing mass in these bins.

        Parameters:
        richness (array-like): Array of richness values.
        log10M_wl (array-like): Array of weak-lensing mass values (log10 scale).
        redshift (array-like): Array of redshift values.
        richness_bins (array-like): Bin edges for richness.
        redshift_bins (array-like): Bin edges for redshift.

        Returns:
        observed_cluster_abundance (2D array): Number of clusters in each bin.
        mean_log10M_wl (2D array): Mean log10 weak-lensing mass in each bin.
        """
        # Compute the 2D histogram for cluster counts
        observed_cluster_abundance, _, _ = np.histogram2d(
            richness, 
            redshift, 
            bins=[ self.richness_bins, self.redshift_bins]
        )

        # Compute the 2D histogram for the sum of log10M_wl
        sum_log10M_wl, _, _ = np.histogram2d(  richness, 
                                               redshift, 
                                               bins=[ self.richness_bins, self.redshift_bins], 
                                               weights=log10M_wl
        )

        # Calculate mean log10M_wl in each bin (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_log10M_wl = np.where( observed_cluster_abundance > 0, 
                                       sum_log10M_wl / observed_cluster_abundance, 
                                       -1 )

        return np.vstack( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten()   
    
# prior = utils.BoxUniform( low = [ 0.05 , 0.5 , 0 , 0.05 , 2 , -1 ] , 
#                       high = [ 1.0 , 1.3 , 2 , 0.5 , 5 , 1 ] )

prior = utils.BoxUniform( low = [ 0.05 , 0.5 , 0 , 0.05 , 2  ] , 
                      high = [ 1.0 , 1.3 , 2 , 0.5 , 5 ] )
    
    
unbinned_simulator = Universe_simulation( 'unbinned_counts' )
print('running long')

unbinned_posterior = infer( unbinned_simulator.run_simulation , 
                            prior, 
                            method = "SNPE", 
                            num_simulations = 300000 , 
                            num_workers = 40 )

with open('unbinned_posterior_long_fine.pkl', "wb") as handle:
    pickle.dump( unbinned_posterior, handle)
    
    
# stacked_simulator = Universe_simulation( 'stacked_counts' )
# print('running no r')


# stacked_posterior = infer( stacked_simulator.run_simulation , 
#                             prior, 
#                             method = "SNPE", 
#                             num_simulations = 300000 , 
#                             num_workers = 40 )

# with open('stacked_posterior_r.pkl', "wb") as handle:
#     pickle.dump( stacked_posterior, handle)
