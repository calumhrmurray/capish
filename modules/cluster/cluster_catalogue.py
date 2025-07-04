import numpy as np

class ClusterCatalogue:
     
    def __init__( self , settings ):
        """
        Initialize the ClusterCatalogue class with the given settings.
        """
        #self.mass_observable_relation = settings["mass_observable_relation"]

    def get_cluster_catalogue( self , halo_catalogue , parameter_set ):
        """
        Generate a cluster catalogue based on the halo catalogue and parameter set.
        """
        # Extract necessary parameters from the parameter set
        mu = halo_catalogue['mu']
        z = halo_catalogue['redshift']

        # Call the mass observable relation function
        return self.mass_observable_relation( mu, z, parameter_set )

    def richness_mass_relation( self,  mu , z , parameter_set  ):
        """
        Returns whatever you want as a cluster catalogue.
        """
        log10Mmin = parameter_set['log10mmin']
        B = parameter_set['b']
        alpha_l = parameter_set['alpha_l']
        beta_l = parameter_set['beta_l']
        z_p = parameter_set['z_p']

        Mmin = 10**log10Mmin
        M1 = 10**( B ) * Mmin
        M = ( np.exp( mu ) * 1e14 )
        mean_l = ( ( M - Mmin ) / ( M1 -  Mmin ) )**alpha_l * ( ( 1 + z ) / ( 1 + z_p ) )**beta_l

        mean_l[ np.logical_or( mean_l < 0, np.isnan(mean_l) ) ] = 0

        return np.log( np.random.poisson( lam = mean_l ) + 1 )

    def mass_observable_relation( self,  mu, z, parameter_set ):

        sigma_l = parameter_set['sigma_l']
        r = parameter_set['r']
        sigma_mwl = parameter_set['sigma_mwl']  

        mean_l = self.richness_mass_relation( mu , z , parameter_set )
        mean_mwl = mu

        cov = [ [ sigma_l**2 , r * sigma_l * sigma_mwl], 
                [r * sigma_l * sigma_mwl, sigma_mwl**2] ]

        total_noise = np.random.multivariate_normal([0, 0], cov=cov, size=len(mean_l))

        # Apply intrinsic noise to mean values
        ln_richness = mean_l + total_noise.T[0]
        lnM_wl = mean_mwl + total_noise.T[1]

        return {
            'richness': np.exp(ln_richness),
            'log10_mwl': np.log10(np.exp(lnM_wl)),
            'redshift': z
        }
