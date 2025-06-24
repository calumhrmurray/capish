import numpy as np

class ClusterCatalogue:
     
    def __init__( self , settings ):
        """
        Initialize the ClusterCatalogue class with the given settings.
        """
        self.mass_observable_relation = settings["mass_observable_relation"]

    def get_cluster_catalogue( self , halo_catalogue , parameter_set ):
        """
        Generate a cluster catalogue based on the halo catalogue and parameter set.
        """
        # Extract necessary parameters from the parameter set
        mu = parameter_set['mu']
        z = parameter_set['z']

        # Call the mass observable relation function
        return self.mass_observable_relation( mu, z, parameter_set )


    def mass_observable_relation(self, mu, z, parameter_set, cosmo ):

        mean_l = self.richness_mass_relation( mu , z , parameter_set )
        mean_mwl = self.c_mwl + self.alpha_mwl * mu

        sampled_l = []
        sampled_mwl = []
        sampled_z = []
        sampled_mu = []

        cov = [[ sigma_l**2 , r * sigma_l * self.sigma_mwl], 
                [r * sigma_l * self.sigma_mwl, self.sigma_mwl**2]]

        total_noise = np.random.multivariate_normal([0, 0], cov=cov, size=len(mean_l))

        # Apply intrinsic noise to mean values
        ln_richness = mean_l + total_noise.T[0]
        lnM_wl = mean_mwl + total_noise.T[1]

        return np.exp( ln_richness ), np.log10( np.exp( lnM_wl ) ), z
