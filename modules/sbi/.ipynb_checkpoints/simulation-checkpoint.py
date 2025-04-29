import numpy as np
import pyccl as ccl
import itertools
import sys
sys.path.append('/pbs/home/c/cmurray/cluster_likelihood/modules/')
import model_halo_abundance


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
        elif summary_statistic == 'des_stacked_counts':
            self.summary_statistic = self.des_stacked_counts
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
        self.z_bins = None
        self.zs = None
        self.mass_grid = None
        self.redshift_grid = None
        self.mass_values = None
        self.redshift_values = None 
        # Bin settings for stacked and unbinned counts
        self.richness_bins = None
        self.redshift_bins = None 
        self.small_log10Mwl_bins = np.arange(12.5, 16, 0.1)
        self.small_richness_bins = np.logspace(np.log10(20), np.log10(300), 15)
        self.small_redshift_bins = np.linspace(0.025, 1.125, 6)
        self.dOmega = None
        self.alpha_mwl = 1
        self.sigma_mwl = 0.25
        self.c_mwl = np.log(1e14)
        self.transfer_function = 'boltzmann_camb'
        self.include_mwl_measurement_errors = False
        self.z_p = 0.3
        self.use_selection_function = False
        self.z_bins_sel = None
        self.l_bins_sel = None
        self.correlation_mass_evolution = False
        self.cme_mu_bins = None
        self.richness_mass_relation = None
        # for the hmf correction, default is set to no correction values
        self.s = 0
        self.q = 1
        self.Mstar = 10**13.8
        #self.omega_b_h2 = 0.02208
        self.Omega_b = 0.048254
    
    def set_richness_mass_relation( self , richness_mass_relation_name ):
        if richness_mass_relation_name == 'power law':
            self.richness_mass_relation = self.power_law
        elif richness_mass_relation_name == 'halo model':
            self.richness_mass_relation = self.halo_model
        elif richness_mass_relation_name == 'constantins model':
            self.richness_mass_relation = self.constantin_power_law
        else:
            print('That mass richness relation is not implemented.')
    
    def set_bins( self, z_bins=None, log10m_bins=None ):
        """
        Set new redshift and mass bins and reinitialize dependent properties.

        Parameters:
            z_bins (np.ndarray): Array of redshift bin edges.
            m_bins (np.ndarray): Array of log10 mass bin edges.
        """
        self.z_bins = z_bins
        self.zs = (self.z_bins[1:] + self.z_bins[:-1]) / 2.

        self.log10ms = log10m_bins
        self.Ms = 10**self.log10ms

        # Reinitialize any dependent properties
        mass_grid, redshift_grid = np.meshgrid(self.log10ms, self.zs)
        self.mass_grid = mass_grid
        self.redshift_grid = redshift_grid
        self.mass_values = self.mass_grid.flatten()
        self.redshift_values = self.redshift_grid.flatten()
        self.dlog10m = log10m_bins[1] - log10m_bins[0]

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
            return self.summary_statistic(richness, log10M_wl, z_clusters)
        #     return torch.tensor( self.summary_statistic(richness, log10M_wl, z_clusters))
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

        Omega_b = self.Omega_b #_h2 / h**2

        print( Om0 , Omega_b , h , sigma8 )
        
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
        mu_clusters, z_clusters = self.get_halo_catalogue( cosmo )

        # Get the observed cluster properties (richness, weak-lensing mass)
        richness, log10M_wl , z_clusters = self.mass_observable_relation( mu_clusters, z_clusters, full_parameter_set , cosmo )
        
        return richness, log10M_wl, z_clusters, mu_clusters
        
    def get_halo_catalogue( self, cosmo, return_Nth=False):

        logm_grid = np.linspace(self.log10ms[0], self.log10ms[-1], 1500)
        dlogm_grid = logm_grid[1] - logm_grid[0]
        logm_grid_center = np.array([(logm_grid[i] + logm_grid[i+1])/2 for i in range(len(logm_grid)-1)])

        clc = model_halo_abundance.ClusterAbundance()
        clc.set_cosmology(cosmo = cosmo, hmd = self.hmf )
        clc.sky_area = self.dOmega
        
        if (self.use_hybrid == False):
            z_grid = np.linspace(self.z_bins[0], self.z_bins[-1], 1000)
            dz_grid = z_grid[1] - z_grid[0]
            z_grid_center = np.array([(z_grid[i] + z_grid[i+1])/2 for i in range(len(z_grid)-1)])
            
            #here, we compute the HMF grid
            clc.compute_multiplicity_grid_MZ(z_grid = z_grid_center, logm_grid = logm_grid_center)
            #we consider using the trapezoidal integral method here, given by int = dx(f(a) + f(b))/2
            hmf_correction = self.hmf_correction(10 ** logm_grid_center, self.Mstar/cosmo['h'], self.s, self.q)
            dN_dzdlogMdOmega_center = clc.dN_dzdlogMdOmega * np.tile(hmf_correction, (len(z_grid_center), 1)).T
            
            if (self.add_SSC == True): 
                clc.compute_halo_bias_grid_MZ(z_grid = z_grid_center, 
                                              logm_grid = logm_grid_center, 
                                              halobiais = self.halobias_fct)
                #generate deltas (log-normal probabilities)
                cov_ln1_plus_delta_SSC = np.log(1 + self.sigmaij_SSC)
                mean = - 0.5 * cov_ln1_plus_delta_SSC.diagonal()
                ln1_plus_delta_SSC = np.random.multivariate_normal(mean=mean , cov=cov_ln1_plus_delta_SSC)
                delta = (np.exp(ln1_plus_delta_SSC) - 1)
                delta_h = clc.halo_biais * delta
                delta_h = np.where(delta_h < -1, -1, delta_h)
                corr = 1 + delta_h
            else: corr = 1

            Omega_z = np.tile(self.dOmega(z_grid_center), (len(z_grid_center), 1)).T
                
            Nobs = np.random.poisson(Omega_z * dN_dzdlogMdOmega_center * dlogm_grid * dz_grid * corr)
            Nobs_flatten = Nobs.flatten()
            Z_grid_center, Logm_grid_center = np.meshgrid(z_grid_center, logm_grid_center)
            Z_grid_center_flatten, Logm_grid_center_flatten = Z_grid_center.flatten(), Logm_grid_center.flatten()

            log10mass = [logm_grid_i for logm_grid_i, count in zip(Logm_grid_center_flatten, Nobs_flatten) for _ in range(count)]
            redshift = [z_grid_i for z_grid_i, count in zip(Z_grid_center_flatten, Nobs_flatten) for _ in range(count)]

            grid = {"N_th": Omega_z * dN_dzdlogMdOmega_center * dlogm_grid * dz_grid, 
                    "z_grid_center":z_grid_center, 
                    "logm_grid_center":logm_grid_center}
        
        elif (self.use_hybrid == True):
            
            Z_edges_hybrid = self.Z_edges_hybrid
            Z_bin_hybrid = [[Z_edges_hybrid[i], Z_edges_hybrid[i+1]] for i in range(len(Z_edges_hybrid)-1)]
            
            #ensure that the redshift grid matches SSC redshift grid values
            z_grid = []
            z_grid.append(Z_edges_hybrid[0])
            for i in range(len(Z_edges_hybrid)-1):
                x = list(np.linspace(Z_edges_hybrid[i], Z_edges_hybrid[i+1], 50))
                z_grid.extend(x[1:-1])
                z_grid.append(Z_edges_hybrid[i+1])
            z_grid = np.array(z_grid)
            dz_grid = z_grid[1] - z_grid[0]
            
            #here, we compute the HMF grid
            clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
            #we consider using the trapezoidal integral method here, given by int = dx(f(a) + f(b))/2
            dN_dzdlogMdOmega_center = (clc.dN_dzdlogMdOmega[:-1] + clc.dN_dzdlogMdOmega[1:]) / 2
            logm_grid_center = np.array([(logm_grid[i] + logm_grid[i+1])/2 for i in range(len(logm_grid)-1)])
            hmf_correction = self.hmf_correction(10**logm_grid_center, self.Mstar/cosmo['h'], self.s, self.q)
            dN_dzdlogMdOmega_center *= np.tile(hmf_correction, (len(z_grid), 1)).T

            if (self.add_SSC == True): 
                clc.compute_halo_bias_grid_MZ(z_grid = z_grid, logm_grid = logm_grid, halobiais = self.halobias_fct)
                halo_bias_center = (clc.halo_biais[:-1] + clc.halo_biais[1:]) / 2
                #generate deltas in redshift bins (log-normal probabilities)
                cov_ln1_plus_delta_SSC = np.log(1 + self.Sij_SSC)
                mean = - 0.5 * cov_ln1_plus_delta_SSC.diagonal()
                ln1_plus_delta_SSC = np.random.multivariate_normal(mean=mean , cov=cov_ln1_plus_delta_SSC)
                delta = (np.exp(ln1_plus_delta_SSC) - 1)

            N_obs = np.zeros([len(Z_bin_hybrid), len(logm_grid_center)])
            N_th = np.zeros([len(Z_bin_hybrid), len(logm_grid_center)])
            log10mass, redshift = [], []
            
            for i, redshift_range in enumerate(Z_bin_hybrid):
                
                mask = (z_grid >= redshift_range[0])*(z_grid <= redshift_range[1])
                dNdm  = self.dOmega(z_grid[mask]) * np.trapz(dN_dzdlogMdOmega_center[:,mask], z_grid[mask], axis=1)
                pdf   = self.dOmega(z_grid[mask]) * dN_dzdlogMdOmega_center[:,mask]
                cumulative = np.cumsum(dz_grid * pdf, axis = 1)  
                
                if self.add_SSC == True:
                    integrand = self.dOmega(z_grid[mask]) * halo_bias_center * dN_dzdlogMdOmega_center
                    bdNdm = np.trapz(integrand[:,mask], z_grid[mask])
                    bias = np.array(bdNdm)/np.array(dNdm)
                    delta_h = bias * delta[i]
                    delta_h = np.where(delta_h < -1, -1, delta_h) #we ensure that deltah = b*delta is > 1
                    corr = (1 + delta_h)
                else: corr = 1
                N_obs[i,:] = np.random.poisson(dNdm * dlogm_grid * corr)
                N_th[i,:] = dNdm * dlogm_grid #we generate the observed count
                N_sample_obs_zbins = N_obs[i,:]
                
                for j in range(len(logm_grid_center)):
                    log10mass.extend(list(np.zeros(int(N_sample_obs_zbins[j]))+logm_grid_center[j])) #masses
                    cumulative_rand = (cumulative[j][-1]-cumulative[j][0])*np.random.random(int(N_sample_obs_zbins[j]))+cumulative[j][0]
                    redshift.extend(list(np.interp(cumulative_rand, cumulative[j], z_grid[mask]))) #redshifts
                    
            grid = {"N_th":N_th, "z_grid":z_grid, "logm_grid":logm_grid, "logm_grid_center": logm_grid_center}

        if return_Nth:
            return grid, log10mass, np.array(redshift)
        else:
            return log10mass, np.array(redshift)
            

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

        # Apply intrinsic noise to mean values
        ln_richness = mean_l + total_noise.T[0]
        lnM_wl = mean_mwl + total_noise.T[1]

        # # Apply observational noise to the richness
        # if self.add_richness_observational_scatter == True:
        #     # background-subtraction noise
        #     delta_bkg = np.random
        #     # projection noise
        #     delta_prj = ( 1 - f_prj ) 

        return np.exp( ln_richness ), np.log10( np.exp( lnM_wl ) ), z
    
    def power_law( self ,  mu , z , Om0, sigma8 , h , w0, wa, alpha_l, c_l, sigma_l, r, beta_l, c_rho, B , log10Mmin  , cosmo ):
        mean_ln_l = c_l + alpha_l * mu + beta_l * np.log( cosmo.h_over_h0(1/(1+z)) / cosmo.h_over_h0(1/(1 + self.z_p) ) )
        # poisson realisation of this values
        ln_l = np.log( np.random.poisson( lam = np.exp( mean_ln_l ) ) )
        return ln_l

    def constantin_power_law( self ,  mu , z , Om0, sigma8 , h , w0, wa, alpha_l, c_l, sigma_l, r, beta_l, c_rho, B , log10Mmin  , cosmo ):
        log10m0 = 14.3
        log10m = np.log10( np.exp( mu ) * 1e14 )
        print( c_l , beta_l , alpha_l )
        mean_ln_l = c_l + beta_l * np.log( ( 1+z ) / (1 + self.z_p ) ) + alpha_l * ( log10m - log10m0 )
        # poisson realisation of this values
        ln_l = np.log( np.random.poisson( lam = np.exp( mean_ln_l )  ) )
        return ln_l
    
    def halo_model( self , mu , z , Om0, sigma8 , h , w0, wa, alpha_l, c_l, sigma_l, r, beta_l, c_rho, B , log10Mmin , cosmo ):
        Mmin = 10**log10Mmin
        M1 = 10**( B ) * Mmin
        M = ( np.exp( mu ) * 1e14 )
        mean_l = ( ( M - Mmin ) / ( M1 -  Mmin ) )**alpha_l * ( ( 1 + z ) / ( 1 + self.z_p ) )**beta_l
        mean_l[ np.logical_or( mean_l < 0, np.isnan(mean_l) ) ] = 0

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
    
    def stacked_counts( self , richness, log10M_wl, redshift , exp =  1):
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
            bins = [ self.richness_bins, self.redshift_bins ]
        )


        # Compute the 2D histogram for the sum of log10M_wl
        sum_log10M_wl, _, _ = np.histogram2d(  richness, 
                                               redshift, 
                                               bins = [ self.richness_bins, self.redshift_bins], 
                                               weights = ( log10M_wl )
        )

        # Calculate mean log10M_wl in each bin (avoid division by zero)
        # we cannot add inverse variance weights here, since we pretend not to know the mass
        # is this fully consistent?
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_log10M_wl = np.where( observed_cluster_abundance > 0, 
                                       sum_log10M_wl / observed_cluster_abundance, 
                                       1 )
            
        # add measurement and systematic uncertainties
        if self.include_mwl_measurement_errors:
            mean_log10M_wl = mean_log10M_wl + np.random.normal( loc = 0.0 , scale = self.mwl_std )
            
        return np.vstack( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten()  
         
    def des_stacked_counts( self , richness, log10M_wl, redshift , exp =  1):
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
            bins = [ self.richness_bins, self.redshift_bins ]
        )


        # Compute the 2D histogram for the sum of log10M_wl
        sum_log10M_wl, _, _ = np.histogram2d(  richness, 
                                               redshift, 
                                               bins = [ self.richness_bins, self.redshift_bins], 
                                               weights = ( 10**log10M_wl )**(1/exp)
        )

        # Calculate mean log10M_wl in each bin (avoid division by zero)
        # we cannot add inverse variance weights here, since we pretend not to know the mass
        # is this fully consistent?
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_log10M_wl = np.where( observed_cluster_abundance > 0, 
                                       sum_log10M_wl / observed_cluster_abundance, 
                                       1 )
            mean_log10M_wl = np.log10( mean_log10M_wl**(exp) )
            
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
    