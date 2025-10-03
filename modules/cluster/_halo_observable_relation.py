from scipy.stats import norm, poisson
import numpy as np


def photometric_redshift(z_true, photoz_params):
    sigma_pz0 = photoz_params
    z_obs = z_true + np.random.randn(len(z_true)) * sigma_pz0 * (1 + z_true)
    return z_obs

class HaloToObservables:
    def __init__(self, config_new):

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

        # ln(lambda) = alpha_lambda + beta_lambda * log10(M - M_min) + gamma_lambda * log10(1 + z)
        # Note: This returns ln(lambda), which gets exponentiated later to get lambda
        ln_lambda = (alpha_lambda + beta_lambda * np.log10(M_term) + gamma_lambda * np.log10(1 + z)) * np.log(10)
        return ln_lambda

    def sigma_obs_relation(self, log10M, z, params_observable_sigma):
        sigma_lambda = params_observable_sigma[0]  # Only use first parameter, others are 0
        return sigma_lambda * np.ones(len(log10M))

    def mean_log10mWL_f(self, log10M, z, params_mWL_mean):

        alpha_mwl, beta_mwl, gamma_mwl = params_mWL_mean
        # Ensure z and log10M are numeric arrays
        z = np.array(z, dtype=float)
        log10M = np.array(log10M, dtype=float)

        # mu_mWL = alpha_mwl + beta_mwl * log10(M) + gamma_mwl * log10(1 + z)
        return alpha_mwl + beta_mwl * log10M + gamma_mwl * np.log10(1 + z)
    
    def sigma_log10mWL_f(self, log10M, z, params_mWL_sigma):

        sigma_mWLgal, sigma_mWLint = params_mWL_sigma
        sigma2 = sigma_mWLgal**2 + sigma_mWLint**2
        return sigma2**.5*np.ones(len(log10M))

    def generate_observables_from_halo(self, log10M, z):

        mean_lnobs = self.mean_obs_relation(log10M, z, self.params_observable_mean)
        sigma_lnobs = self.sigma_obs_relation(log10M, z, self.params_observable_sigma)
        mean_log10mWL = self.mean_log10mWL_f(log10M, z, self.params_mWL_mean)
        sigma_log10mWL = self.sigma_log10mWL_f(log10M, z, self.params_mWL_sigma)
        rho = self.rho_obs_mWL

        if self.which_mass_richness_rel!='GPC':
        
            if self.which_mass_richness_rel=='Gauss+Poiss-corr':
                mean_lnobs = mean_lnobs - 0.5 * np.exp(-mean_lnobs+0.5*sigma_lnobs**2) - (1/12)*np.exp(-2*mean_lnobs+2*sigma_lnobs**2)
            if self.which_mass_richness_rel=='Gauss':
                mean_lnobs = mean_lnobs
            sigma_lnobs2 = sigma_lnobs**2
            #add poisson noise
            sigma_lnobs2 = sigma_lnobs2 + (np.exp(mean_lnobs)-1)/np.exp(2*mean_lnobs)
            sigma_lnobs = sigma_lnobs2**.5
    
            lnobs_noise = np.random.normal(loc=0, scale=sigma_lnobs)
            lnobs = mean_lnobs + lnobs_noise
    
            cond_mean_log10mWL = mean_log10mWL + rho * (sigma_log10mWL / sigma_lnobs) * (lnobs - mean_lnobs)
            cond_sigma_log10mWL = sigma_log10mWL * np.sqrt(1 - rho**2)
            log10Mwl = cond_mean_log10mWL + np.random.normal(loc=0, scale=cond_sigma_log10mWL)

        elif self.which_mass_richness_rel=='GPC':
            #only if rho == 0, else not relevant
            n_halo = len(z)
            lnint = mean_lnobs + np.random.randn(n_halo) * sigma_lnobs
            lnobs = np.log(np.random.poisson(lam=np.exp(lnint)))
            log10Mwl = mean_log10mWL + np.random.randn(n_halo) * sigma_log10mWL

        if self.add_photoz:
            
            z = photometric_redshift(z, self.photoz_params)
        
        return np.exp(lnobs), log10Mwl, z

    
    