from scipy.stats import norm, poisson
import numpy as np


def photometric_redshift(z_true, photoz_params):
    sigma_pz0 = photoz_params
    z_obs = z_true + np.random.randn(len(z_true)) * sigma_pz0 * (1 + z_true)
    return z_obs

class HaloToObservables:
    def __init__(self, config_new, sigma_log10Mwl_gal_interp = None):

        parameters = config_new['parameters']
        log10M_min = float(parameters['log10M_min'])
        z0 = float(parameters['z0'])
        params_observable_mean = [float(parameters['alpha_lambda']), float(parameters['beta_lambda']), float(parameters['gamma_lambda'])]
        params_observable_sigma = [float(parameters['sigma_lambda']), 0.0, 0.0]  # Only sigma_lambda used
        params_mWL_mean = [float(parameters['alpha_mwl']), float(parameters['beta_mwl']), float(parameters['gamma_mwl'])]
        params_mWL_sigma = [float(parameters['sigma_Mwl_gal']), float(parameters['sigma_Mwl_int'])]
        which_mass_richness_rel = config_new['cluster_catalogue.mass_observable_relation']['which_relation']
        add_correction_to_mean_log10Mwl = True if config_new['cluster_catalogue']['add_correction_to_mean_log10Mwl']=='True' else False
        params_rho_mWL = [float(parameters['rho_0']), float(parameters['rho_A']), float(parameters['rho_alpha']), float(parameters['rho_log10m0'])]
        add_photoz = True if config_new['cluster_catalogue']['add_photometric_redshift']=='True' else False
        photoz_params = float(config_new['cluster_catalogue.photometric_redshift']['sigma_z0'])
    
        self.log10M_min = log10M_min
        self.z0 = z0
        self.use_theory_for_sigma_Mwl_gal = config_new['cluster_catalogue']['theory_sigma_Mwl_gal']
        self.add_correction_to_mean_log10Mwl = add_correction_to_mean_log10Mwl
        self.sigma_log10Mwl_gal_interp = sigma_log10Mwl_gal_interp # = None if self.use_theory_for_sigma_Mwl_gal == 'False'
        self.gaussian_lensing_variable = config_new['cluster_catalogue']['gaussian_lensing_variable']
        self.params_observable_mean = params_observable_mean
        self.params_observable_sigma = params_observable_sigma
        self.params_mWL_mean = params_mWL_mean
        self.params_mWL_sigma = params_mWL_sigma
        self.rho_obs_mWL_params = params_rho_mWL
        self.add_photoz = add_photoz
        self.photoz_params = photoz_params
        self.which_mass_richness_rel = which_mass_richness_rel

    def mean_obs_relation_DES(self, log10M, z, params_observable_mean):
        alpha_lambda, beta_lambda, gamma_lambda = params_observable_mean
        # Ensure all are float
        alpha_lambda = float(alpha_lambda)
        beta_lambda = float(beta_lambda)
        gamma_lambda = float(gamma_lambda)

        # Ensure z and log10M are numeric arrays
        z = np.array(z, dtype=float)
        log10M = np.array(log10M, dtype=float)

        # Convert log10M to M, subtract M_min, then back to log10
        M = 10 ** log10M
        M_term = M - 10 ** self.log10M_min

        # Ensure M_term is positive to avoid log of negative numbers
        M_term = np.maximum(M_term, 1e10)  # Set minimum value to avoid log issues

        # Note: This returns ln(lambda), which gets exponentiated later to get lambda
        ln_lambda = np.log(10) * (alpha_lambda + beta_lambda * np.log10(M_term) + gamma_lambda * np.log10(1 + z))
        return ln_lambda

    def mean_obs_relation_power_law(self, log10M, z, params_observable_mean):
        alpha_lambda, beta_lambda, gamma_lambda = params_observable_mean
        # Ensure all are float
        alpha_lambda = float(alpha_lambda)
        beta_lambda = float(beta_lambda)
        gamma_lambda = float(gamma_lambda)

        z = np.array(z, dtype=float)
        log10M = np.array(log10M, dtype=float)
        
        ln_lambda = alpha_lambda + beta_lambda * (log10M - self.log10M_min) + gamma_lambda * np.log((1 + z)/(1 + self.z0))
        return ln_lambda

    def sigma_obs_relation(self, log10M, z, params_observable_sigma):
        sigma_lambda = params_observable_sigma[0]  # Only use first parameter, others are 0
        return sigma_lambda * np.ones(len(log10M))

    def mean_log10mWL_f(self, log10M, z, params_mWL_mean):

        alpha_mwl, beta_mwl, gamma_mwl = params_mWL_mean
        # Ensure z and log10M are numeric arrays
        z = np.array(z, dtype=float)
        log10M = np.array(log10M, dtype=float)

        return alpha_mwl + beta_mwl * log10M + gamma_mwl * np.log10(1 + z)

    def rho_mWL_f(self, log10m, z, rho_params):
        rho_0, rho_A, rho_alpha, rho_log10m0 = rho_params
        return rho_0 + rho_A * np.exp(- rho_alpha * (log10m - rho_log10m0))

    def generate_observables_from_halo(self, log10M, z):
        """
        Generate mock observables for a halo given its mass and redshift.
    
        Parameters
        ----------
        log10M : float or array-like
            True halo mass in log10(M/M_sun).
        z : float or array-like
            Halo redshift.
        WL_random : str, optional
            Method for generating weak-lensing mass:
              - 'log10Mwl': use log10-space conditional distribution.
              - 'Mwl' (default): use linear mass-space conditional distribution.
    
        Returns
        -------
        lnobs_exp : float or array-like
            Realization of the observable (exp(lnobs)) with intrinsic scatter.
        log10Mwl : float or array-like
            Weak-lensing mass estimate in log10(M/M_sun), including correlated scatter.
        z_out : float or array-like
            Redshift of the halo (possibly perturbed by photometric scatter).

        """
        # Compute mean and scatter for the observable
        if self.which_mass_richness_rel == 'DES':
            mean_lnobs = self.mean_obs_relation_DES(log10M, z, self.params_observable_mean)
        elif self.which_mass_richness_rel == 'power_law':
            mean_lnobs = self.mean_obs_relation_power_law(log10M, z, self.params_observable_mean)
            
        sigma_lnobs = self.sigma_obs_relation(log10M, z, self.params_observable_sigma)
            
        if self.use_theory_for_sigma_Mwl_gal=='False':
            sigma_mWLgal, sigma_mWLint = self.params_mWL_sigma
            sigma2 = sigma_mWLgal**2 + sigma_mWLint**2
            sigma_log10mWL = sigma2 ** .5
        else:
            sigma_mWLgal, sigma_mWLint = self.params_mWL_sigma
            sigma2 = self.sigma_log10Mwl_gal_interp(log10M, z) ** 2 + sigma_mWLint ** 2
            sigma_log10mWL = sigma2 ** .5

        # Compute mean and scatter for weak-lensing mass
        mean_log10mWL = self.mean_log10mWL_f(log10M, z, self.params_mWL_mean)
        if self.add_correction_to_mean_log10Mwl: #it centers the mass
            #this ensures that <Mwl> = 10^(mean_logmWL)
            #this is important for large values of sigma_mWLgal (like using the model)
            mean_logmWL = mean_log10mWL * np.log(10)
            mean_logmWL = mean_logmWL - 0.5 * (sigma_log10mWL * np.log(10)) ** 2
            mean_log10mWL = mean_logmWL/np.log(10)
    
        rho = self.rho_mWL_f(log10M, z, self.rho_obs_mWL_params)
    
        # Include extra term in observable scatter (optional)
        sigma_lnobs2 = sigma_lnobs ** 2
        sigma_lnobs2 = sigma_lnobs2 + np.exp(-mean_lnobs)
        sigma_lnobs = np.sqrt(sigma_lnobs2)
    
        # Draw observable from Gaussian scatter
        lnobs_noise = np.random.normal(loc=0, scale=sigma_lnobs)
        lnobs = mean_lnobs + lnobs_noise
    
        # Sample weak-lensing mass conditional on lnobs
        if self.gaussian_lensing_variable == 'log10Mwl':
            cond_mean_log10mWL = mean_log10mWL + rho * (sigma_log10mWL / sigma_lnobs) * (lnobs - mean_lnobs)
            cond_sigma_log10mWL = sigma_log10mWL * np.sqrt(1 - rho**2)
            log10Mwl = cond_mean_log10mWL + np.random.normal(loc=0, scale=cond_sigma_log10mWL)
    
        elif self.gaussian_lensing_variable == 'Mwl':
            #bettet for large scatter, otherwise will shift the mean mass
            mean_mWL = 10 ** mean_log10mWL
            m = 10 ** log10M
            sigma_mWL = sigma_log10mWL * (np.log(10) * m) #conversion
            cond_mean_mWL = mean_mWL + rho * (sigma_mWL / sigma_lnobs) * (lnobs - mean_lnobs)
            cond_sigma_mWL = sigma_mWL * np.sqrt(1 - rho**2)
            Mwl = cond_mean_mWL + np.random.normal(loc=0, scale=cond_sigma_mWL)
            Mwl = np.maximum(Mwl, 1e-5) #ensure that the mass is positive
            log10Mwl = np.log10(Mwl)
    
        # Perturb redshift if photometric scatter is enabled
        if self.add_photoz:
            z = photometric_redshift(z, self.photoz_params)
    
        # Return observable in linear space, weak-lensing mass, and redshift
        return np.exp(lnobs), log10Mwl, z