from scipy.stats import norm, poisson
import numpy as np


def photometric_redshift(z_true, photoz_params):
    sigma_pz0 = photoz_params
    z_obs = z_true + np.random.randn(len(z_true)) * sigma_pz0 * (1 + z_true)
    return z_obs

class HaloToObservables:
    def __init__(self, config_new, sigma_log10mWL_model = None):

        parameters = config_new['parameters']
        M_min = float(parameters['M_min'])
        params_observable_mean = [float(parameters['alpha_lambda']), float(parameters['beta_lambda']), float(parameters['gamma_lambda'])]
        params_observable_sigma = [float(parameters['sigma_lambda']), 0.0, 0.0]  # Only sigma_lambda used
        params_mWL_mean = [float(parameters['alpha_mwl']), float(parameters['beta_mwl']), float(parameters['gamma_mwl'])]
        params_mWL_sigma = [float(parameters['sigma_Mwl_gal']), float(parameters['sigma_Mwl_int'])]
        rho_obs_mWL = float(parameters['rho'])
        which_mass_richness_rel = config_new['cluster_catalogue.mass_observable_relation']['which_relation']

        add_photoz = True if config_new['cluster_catalogue']['add_photometric_redshift']=='True' else False
        photoz_params = float(config_new['cluster_catalogue.photometric_redshift']['sigma_z0'])
    
        self.M_min = M_min
        self.theory_sigma_Mwl = config_new['cluster_catalogue']['theory_sigma_Mwl']
        self.sigma_log10mWL_model = sigma_log10mWL_model
        self.params_observable_mean = params_observable_mean
        self.params_observable_sigma = params_observable_sigma
        self.params_mWL_mean = params_mWL_mean
        self.params_mWL_sigma = params_mWL_sigma
        self.rho_obs_mWL = rho_obs_mWL
        self.add_photoz = add_photoz
        self.photoz_params = photoz_params
        self.which_mass_richness_rel = which_mass_richness_rel

    def mean_obs_relation(self, log10M, z, params_observable_mean):
        alpha_lambda, beta_lambda, gamma_lambda = params_observable_mean
        # Ensure all are float
        alpha_lambda = float(alpha_lambda)
        beta_lambda = float(beta_lambda)
        gamma_lambda = float(gamma_lambda)

        # Ensure z and log10M are numeric arrays
        z = np.array(z, dtype=float)
        log10M = np.array(log10M, dtype=float)

        # Convert log10M to M, subtract M_min, then back to log10
        M = 10**log10M
        M_term = M - self.M_min

        # Ensure M_term is positive to avoid log of negative numbers
        M_term = np.maximum(M_term, 1e10)  # Set minimum value to avoid log issues

        # Note: This returns ln(lambda), which gets exponentiated later to get lambda
        ln_lambda = alpha_lambda + beta_lambda * np.log10(M_term) + gamma_lambda * np.log10(1 + z)
        return ln_lambda * np.log(10)

    def sigma_obs_relation(self, log10M, z, params_observable_sigma):
        sigma_lambda = params_observable_sigma[0]  # Only use first parameter, others are 0
        return sigma_lambda * np.ones(len(log10M))

    def mean_log10mWL_f(self, log10M, z, params_mWL_mean):

        alpha_mwl, beta_mwl, gamma_mwl = params_mWL_mean
        # Ensure z and log10M are numeric arrays
        z = np.array(z, dtype=float)
        log10M = np.array(log10M, dtype=float)

        return alpha_mwl + beta_mwl * log10M + gamma_mwl * np.log10(1 + z)
    
    def sigma_log10mWL_f(self, log10M, z, params_mWL_sigma):

        sigma_mWLgal, sigma_mWLint = params_mWL_sigma
        sigma2 = sigma_mWLgal**2 + sigma_mWLint**2
        return sigma2**.5*np.ones(len(log10M))

    def generate_observables_from_halo(self, log10M, z, WL_random='Mwl'):
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
        mean_lnobs = self.mean_obs_relation(log10M, z, self.params_observable_mean)
        sigma_lnobs = self.sigma_obs_relation(log10M, z, self.params_observable_sigma)
    
        # Compute mean and scatter for weak-lensing mass
        mean_log10mWL = self.mean_log10mWL_f(log10M, z, self.params_mWL_mean)
        if self.theory_sigma_Mwl == 'False':
            sigma_log10mWL = self.sigma_log10mWL_f(log10M, z, self.params_mWL_sigma)
        else:
            sigma_log10mWL = self.sigma_log10mWL_model(log10M, z)
    
        rho = self.rho_obs_mWL
    
        # Include extra term in observable scatter (optional)
        sigma_lnobs2 = sigma_lnobs**2
        sigma_lnobs2 = sigma_lnobs2 + np.exp(-mean_lnobs)
        sigma_lnobs = np.sqrt(sigma_lnobs2)
    
        # Draw observable from Gaussian scatter
        lnobs_noise = np.random.normal(loc=0, scale=sigma_lnobs)
        lnobs = mean_lnobs + lnobs_noise
    
        # Sample weak-lensing mass conditional on lnobs
        if WL_random == 'log10Mwl':
            cond_mean_log10mWL = mean_log10mWL + rho * (sigma_log10mWL / sigma_lnobs) * (lnobs - mean_lnobs)
            cond_sigma_log10mWL = sigma_log10mWL * np.sqrt(1 - rho**2)
            log10Mwl = cond_mean_log10mWL + np.random.normal(loc=0, scale=cond_sigma_log10mWL)
    
        elif WL_random == 'Mwl':
            #bettet for large scatter, otherwise will shift the mean mass
            mean_mWL = 10 ** mean_log10mWL
            sigma_mWL = sigma_log10mWL * np.log(10) * log10M
            cond_mean_mWL = mean_mWL + rho * (sigma_mWL / sigma_lnobs) * (lnobs - mean_lnobs)
            cond_sigma_mWL = sigma_mWL * np.sqrt(1 - rho**2)
            Mwl = cond_mean_mWL + np.random.normal(loc=0, scale=cond_sigma_mWL)
            log10Mwl = np.log10(Mwl)
    
        # Perturb redshift if photometric scatter is enabled
        if self.add_photoz:
            z = photometric_redshift(z, self.photoz_params)
    
        # Return observable in linear space, weak-lensing mass, and redshift
        return np.exp(lnobs), log10Mwl, z


    
    