import numpy as np
import pyccl as ccl
import itertools

class Universe_simulation:
    
    def __init__(self, summary_statistic, variable_params, fixed_params=None, for_simulate_for_sbi=False):
        """
        Initialize the Universe_simulation class.
        """
        # Set the summary statistic function
        if summary_statistic == 'stacked_counts':
            self.summary_statistic = self.stacked_counts
        elif summary_statistic == 'unbinned_counts':
            self.summary_statistic = self.unbinned_counts
        elif summary_statistic == 'stacked_counts_wonky_bins':
            self.summary_statistic = self.stacked_counts_wonky_bins
        else:
            raise ValueError(f'Unknown summary statistic: {summary_statistic}')

        # Define available parameters and their default values
        self.available_params = {
            'omega_m': 0.3,
            'sigma_8': 0.8,
            'ln_1010_As': 4.21,
            'h': 0.7,
            'w_0': -1,
            'w_a': 0,
            'alpha': 0.8,
            'c': 3,
            'sigma': 0.3,
            'r': 0.0,
            'beta': 0,
            'c_rho': 0.0,
            'B':0,
            'log10Mmin':0,
        }

        # Specify the variable and fixed parameters
        self.variable_params = variable_params
        self.fixed_params = fixed_params if fixed_params else {}

        # Other simulation setup
        self.selection_richness = 0
        self.for_simulate_for_sbi = for_simulate_for_sbi
        # HMF properties
        self.dlog10m = 0.01
        self.log10ms = np.arange( 13., 16, self.dlog10m )
        self.Ms = 10**self.log10ms
        self.hmf = ccl.halos.MassFuncTinker10(mass_def='200m')
        # Create mass and redshift grids
        self.z_bins = np.arange(0, 1.2, 0.05)
        self.zs = (self.z_bins[1:] + self.z_bins[:-1]) / 2.
        mass_grid, redshift_grid = np.meshgrid( self.log10ms, self.zs )
        self.mass_grid = mass_grid
        self.redshift_grid = redshift_grid
        self.mass_values = self.mass_grid.flatten()
        self.redshift_values = self.redshift_grid.flatten()
        # Bin settings for stacked and unbinned counts
        self.richness_bins = np.logspace(np.log10(20), np.log10(300), 15)
        self.redshift_bins = np.linspace(0.025, 1.125, 6)
        self.small_log10Mwl_bins = np.arange(12.5, 16, 0.1)
        self.small_richness_bins = np.logspace(np.log10(20), np.log10(300), 15)
        self.small_redshift_bins = np.linspace(0.025, 1.125, 6)
        self.dOmega = 0.5 * 4 * np.pi
        self.alpha_mwl = 1
        self.sigma_mwl = 0.3
        self.c_mwl = np.log(1e14)
        self.transfer_function = 'boltzmann_camb'
        self.include_mwl_measurement_errors = False
        self.z_p = 0.3
        self.use_selection_function = False
        self.z_bins_sel = None
        self.l_bins_sel = None
        self.selection_function = None
        self.correlation_mass_evolution = False
        self.cme_mu_bins = None
        self.richness_mass_relation = self.power_law
        # for the hmf correction, default is set to no correction values
        self.s = 0
        self.q = 1
        self.Mstar = 10**13.8
        self.omega_b_h2 = 0.02208
    
    def set_richness_mass_relation( self , richness_mass_relation_name ):
        if richness_mass_relation_name == 'power law':
            self.richness_mass_relation = self.power_law
        elif richness_mass_relation_name == 'halo model':
            self.richness_mass_relation = self.halo_model
        else:
            print('That mass richness relation is not implemented.')
    
    def set_bins( self, z_bins=None, log10m_bins=None ):
        """
        Set new redshift and mass bins and reinitialize dependent properties.

        Parameters:
            z_bins (np.ndarray): Array of redshift bin edges.
            m_bins (np.ndarray): Array of log10 mass bin edges.
        """
        if z_bins is not None:
            self.z_bins = z_bins
            self.zs = (self.z_bins[1:] + self.z_bins[:-1]) / 2.

        if log10m_bins is not None:
            self.log10ms = log10m_bins
            self.Ms = 10**self.log10ms

        # Reinitialize any dependent properties
        mass_grid, redshift_grid = np.meshgrid(self.log10ms, self.zs)
        self.mass_grid = mass_grid
        self.redshift_grid = redshift_grid
        self.mass_values = self.mass_grid.flatten()
        self.redshift_values = self.redshift_grid.flatten()

    def _get_parameter_set(self, param_values):
        """
        Create the full parameter set by combining fixed and variable parameters.
        """
        # Start with default parameter values
        parameter_set = self.available_params.copy()

        # Assign the passed variable parameters to their corresponding keys
        for i, param in enumerate( self.variable_params ):
            parameter_set[param] = float(param_values[i])  # Ensure float type

        # Overwrite with any fixed parameters
        parameter_set.update( self.fixed_params)

        # Return the ordered list of parameters
        return [ parameter_set[p] for p in self.available_params ]

    def run_simulation(self, param_values):
        """
        Run the simulation using the variable parameters provided.
        """
        # Get the full parameter set (both variable and fixed parameters)
        full_parameter_set = self._get_parameter_set( param_values )

        # Run the core simulation
        richness, log10M_wl, z_clusters, _  = self._run_simulation( full_parameter_set )

        # Return result in a format compatible with SBI
        if self.for_simulate_for_sbi:
            return torch.tensor(self.summary_statistic(richness, log10M_wl, z_clusters))
        else:
            return self.summary_statistic(richness, log10M_wl, z_clusters)

    def _run_simulation(self, full_parameter_set ):
        """
        Core function to simulate cluster catalogs and selection function.
        """
        # make cosmo
        Om0, sigma8 , ln_1010_As , h, w0, wa, alpha_l, c_l, sigma_l, r, beta_l, c_rho, B , log10Mmin = full_parameter_set

        # Ensure that parameters are native Python floats (not PyTorch tensors)
        Om0 = float(Om0)
        sigma8 = float(sigma8)
        h = float(h)
        w0 = float(w0)
        wa = float(wa)

        Omega_b = self.omega_b_h2 / h**2
        
        cosmo_params = {
                        'Omega_c': Om0 - Omega_b, 
                        'Omega_b': Omega_b,
                        'h': h,    
                        'n_s': 0.96,
                        'sigma8': sigma8,   
                        'Omega_k': 0.0 ,
                        'matter_power_spectrum' : 'linear',
                        'transfer_function': self.transfer_function,
                        'w0': w0,
                        'wa': wa,
                        'extra_parameters':{"camb": {"dark_energy_model": "ppf"}}
                    }

        # Create the CCL Cosmology object once
        cosmo = ccl.Cosmology( **cosmo_params )
        
        # Get the latent cluster properties (mu_clusters, z_clusters)
        mu_clusters, z_clusters = self.get_cluster_catalogue( cosmo )

        # Get the observed cluster properties (richness, weak-lensing mass)
        richness, log10M_wl , z_clusters = self.mass_observable_relation( mu_clusters, z_clusters, full_parameter_set , cosmo )

        # Apply selection function
        selection = richness > self.selection_richness
        
        return richness[selection], log10M_wl[selection], z_clusters[selection], mu_clusters[selection]
        
    def get_cluster_catalogue( self, cosmo ):

        z_bin_centers = (self.z_bins[:-1] + self.z_bins[1:]) / 2.0
        scale_factor_bins = 1/( self.z_bins + 1 )
        scale_factors = 1 / (z_bin_centers + 1)

        # Compute comoving volumes only once
        dV = cosmo.comoving_volume_element( scale_factors ) 
        da = scale_factor_bins[:-1] - scale_factor_bins[1:]
        
        cluster_abundance = []

        for i, a in enumerate(scale_factors):
            # Calculate halo mass function for the current redshift (as scalar `a`)
            # Mstar units were Msun/h so we need to divide to get it into sensible units
            dndlog10M = self.hmf( cosmo, self.Ms, a ) * self.hmf_correction( self.Ms , self.Mstar / cosmo['h'] , self.s , self.q )

            # Compute counts in each bin
            # I don't know why I had a factor of a[i] in here
            counts_per_bin = np.random.poisson( dndlog10M * dV[i] * self.dlog10m * self.dOmega * da[i] )
            cluster_abundance.append(counts_per_bin)

        cluster_abundance = np.array(cluster_abundance).flatten()

        # Use np.repeat to create the catalog based on counts in cluster_abundance
        cat_mass = np.repeat( self.mass_values, cluster_abundance )
        cat_redshift = np.repeat( self.redshift_values, cluster_abundance )
        # should we correct for little h here?
        cat_mu = np.log( 10 ** cat_mass / 1e14  )

        return cat_mu, cat_redshift

    def hmf_correction( self , M , Mstar , s , q ):
        return s * np.log10( M / Mstar ) + q

    def mass_observable_relation(self, mu, z, full_parameter_set, cosmo ):
        Om0, sigma8 , ln_1010_As , h , w0, wa, alpha_l, c_l, sigma_l, r, beta_l, c_rho, B , log10Mmin = full_parameter_set

        # Ensure that parameters are native Python floats (not PyTorch tensors)
        # I am not sure that this is needed actually
        alpha_l = float(alpha_l)
        sigma_l = float(sigma_l)
        c_l = float(c_l)
        r = float(r)

        mean_l = self.richness_mass_relation( mu , z , Om0, sigma8 , h , w0, wa, alpha_l, c_l, sigma_l, r, beta_l, c_rho, B , log10Mmin , cosmo )
        mean_mwl = self.c_mwl + self.alpha_mwl * mu

        sampled_l = []
        sampled_mwl = []
        sampled_z = []
        sampled_mu = []

        # Selection function
        if self.use_selection_function:
            for i in np.arange(0, len(self.z_bins_sel[1:])):
                for j in np.arange(0, len(self.l_bins_sel[1:])):
                    bin_indices = np.where((z > self.z_bins_sel[i]) &
                                           (z < self.z_bins_sel[i+1]) &
                                           (mean_l > self.l_bins_sel[j]) &
                                           (mean_l < self.l_bins_sel[j+1]))[0]
                    # Randomly sample indices
                    sampled_indices = np.random.choice(bin_indices, 
                                                       int(self.selection_function[i][j] * len(mu[bin_indices])), 
                                                       replace=False)
                    sampled_l.append( mean_l[sampled_indices] )
                    sampled_mwl.append( mean_mwl[sampled_indices] )
                    sampled_z.append( z[sampled_indices] )
                    sampled_mu.append( mu[sampled_indices] )

            mean_l = np.concatenate( sampled_l )
            mean_mwl = np.concatenate( sampled_mwl )
            z = np.concatenate( sampled_z )
            mu = np.concatenate( sampled_mu )


        # Initialize noise array
        total_noise = np.zeros( (len( mu ) , 2 ) )

        # Mass-evolution correlation
        if self.correlation_mass_evolution:

            for i in np.arange( 0 , len( self.cme_mu_bins[:-1] ) ):
                
                bin_indices = np.where( ( mu > self.cme_mu_bins[i] ) & ( mu < self.cme_mu_bins[i+1] ))[0]

                # Calculate mass-evolution-dependent covariance for each bin
                r_mu = r #* np.mean( mus[bin_indices] ) + c_rho
                
                cov = [[ sigma_l**2, r_mu * sigma_l * self.sigma_mwl ], 
                       [ r_mu * sigma_l * self.sigma_mwl, self.sigma_mwl**2 ]]
                
                noise = np.random.multivariate_normal( [0, 0], 
                                                       cov = cov, 
                                                       size=len( mu[bin_indices] ) )

                # Add noise to corresponding indices
                total_noise[bin_indices] = noise

        else:
            # Default covariance without mass evolution
            cov = [[sigma_l**2 , r * sigma_l * self.sigma_mwl], 
                   [r * sigma_l * self.sigma_mwl, self.sigma_mwl**2]]

            total_noise = np.random.multivariate_normal([0, 0], cov=cov, size=len(mean_l))

        # Apply noise to mean values
        ln_richness = mean_l + total_noise.T[0]
        lnM_wl = mean_mwl + total_noise.T[1]

        return np.exp(ln_richness), np.log10(np.exp(lnM_wl)), z
    
    def power_law( self ,  mu , z , Om0, sigma8 , h , w0, wa, alpha_l, c_l, sigma_l, r, beta_l, c_rho, B , log10Mmin  , cosmo ):
        mean_ln_l = c_l + alpha_l * mu + beta_l * np.log( cosmo.h_over_h0(1/(1+z)) / cosmo.h_over_h0(1/(1 + self.z_p) ) )
        # poisson realisation of this values
        ln_l = np.log( np.random.poisson( lam = np.exp( mean_ln_l ) ) )
        return ln_l
    
    def halo_model( self , mu , z , Om0, sigma8 , h , w0, wa, alpha_l, c_l, sigma_l, r, beta_l, c_rho, B , log10Mmin , cosmo ):
        # unfortunately we need to add in the h dependence
        Mmin = 10**log10Mmin / h
        M1 = 10**( B ) * Mmin
        M = ( np.exp( mu ) * 1e14 )
        mean_l = ( ( M - Mmin ) / ( M1 -  Mmin ) )**alpha_l * ( ( 1 + z ) / ( 1 + self.z_p ) )**beta_l
        mean_l[np.logical_or(mean_l < 0, np.isnan(mean_l))] = 0
        return np.log( np.random.poisson( lam = mean_l ) + 1 )

    def three_dim_counts(self, richness, log10M_wl, z_clusters ):
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
        # the lensing mass counts are inverse variance weighted
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
                                               bins = [ self.richness_bins, self.redshift_bins], 
                                               weights = ( 10**log10M_wl )**(1/3)
        )

        # Calculate mean log10M_wl in each bin (avoid division by zero)
        # we cannot add inverse variance weights here, since we pretend not to know the mass
        # is this fully consistent?
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_log10M_wl = np.where( observed_cluster_abundance > 0, 
                                       sum_log10M_wl / observed_cluster_abundance, 
                                       1 )
            mean_log10M_wl = np.log10( mean_log10M_wl**3 )
            
        # add measurement and systematic uncertainties
        if self.include_mwl_measurement_errors:
            mean_log10M_wl = mean_log10M_wl + np.random.normal( loc = 0.0 , scale = self.mwl_std )
            
        return np.vstack( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten()  
    
    def stacked_counts_wonky_bins( self , richness, log10M_wl, redshift ):
        """
        Calculate the number of clusters in bins of cluster richness and redshift,
        and calculate the mean cluster weak-lensing mass in these bins.
        
        For KiDs where the richness bins change with redshift for the mass estimates 
        and they are defined completely different for the cluster counts

        Parameters:
        richness (array-like): Array of richness values.
        log10M_wl (array-like): Array of weak-lensing mass values (log10 scale).
        redshift (array-like): Array of redshift values.
        richness_bins (array-like): Bin edges for richness.
        redshift_bins (array-like): Bin edges for redshift.

        Returns:
        observed_cluster_abundance (1D array): Number of clusters in each bin.
        mean_log10M_wl (1D array): Mean log10 weak-lensing mass in each bin.
        """
        
        
        # counts....
        # Compute the 2D histogram for cluster counts
        observed_cluster_abundance, _, _ = np.histogram2d(
            richness, 
            redshift, 
            bins=[ self.richness_bins, self.redshift_bins ]
        )
        
        observed_cluster_abundance = observed_cluster_abundance.flatten()
        
        mean_log10M_wl = []
        
        # mass measurements...
        for idx_bin in np.arange( 0 , len( self.wl_redshift_bins[:-1] ) ):
            
            redshift_selection = ( redshift > self.wl_redshift_bins[idx_bin] ) & ( redshift < self.wl_redshift_bins[idx_bin+1] )
            
            in_observed_cluster_abundance, _ = np.histogram( richness[ redshift_selection ] , 
                                                             bins = self.wl_richness_bins[ idx_bin ] )
            
            sum_log10M_wl, _ = np.histogram( richness[ redshift_selection ] ,
                                             weights = ( 10**log10M_wl[ redshift_selection ] )**(1/3) ,
                                             bins = self.wl_richness_bins[idx_bin] )
            
            with np.errstate(divide='ignore', invalid='ignore'):
                in_mean_log10M_wl = np.where( in_observed_cluster_abundance > 0, 
                                              sum_log10M_wl / in_observed_cluster_abundance, 
                                              -1 )
                in_mean_log10M_wl = np.log10( in_mean_log10M_wl**3 )
            

            # add measurement and systematic uncertainties
            if self.include_mwl_measurement_errors:
                in_mean_log10M_wl = in_mean_log10M_wl + np.random.normal( loc = 0.0 , scale = self.mwl_std[idx_bin] )
                
            mean_log10M_wl.append( in_mean_log10M_wl )

        mean_log10M_wl = np.array( list( itertools.chain.from_iterable( mean_log10M_wl ) ) )
                    
        return np.concatenate( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten()  
    
    def old_stacked_counts( self , richness, log10M_wl, redshift ):
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
                                               weights=( 10**log10M_wl )
        )

        # Calculate mean log10M_wl in each bin (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_log10M_wl = np.where( observed_cluster_abundance > 0, 
                                       sum_log10M_wl / observed_cluster_abundance, 
                                       -1 )
            
            mean_log10M_wl = np.log10( mean_log10M_wl )
            

        return np.vstack( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten()  
    