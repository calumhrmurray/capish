from scipy.stats import norm, poisson
import numpy as np


def photometric_redshift(z_true, photoz_params):
    sigma_pz0 = photoz_params
    z_obs = z_true + np.random.randn(len(z_true)) * sigma_pz0 * (1 + z_true)
    return z_obs

class HaloToObservables:
    def __init__(self, config_new):

        parameters = config_new['parameters']
        pivot_obs_z0 = float(parameters['pivot_obs_z0'])
        pivot_obs_log10m0 = float(parameters['pivot_obs_log10m0'])
        params_observable_mean = [float(parameters['mu_0_lambda']), float(parameters['mu_z_lambda']), float(parameters['mu_m_lambda'])]
        params_observable_sigma = [float(parameters['sigma_lambda']), 0.0, 0.0]  # Only sigma_lambda used
        params_observable_mean = [float(pivot_obs_log10m0), float(pivot_obs_z0)] + params_observable_mean
        params_observable_sigma = [float(pivot_obs_log10m0), float(pivot_obs_z0)] + params_observable_sigma
        params_mWL_mean = [float(parameters['mu_0_Mwl']), float(parameters['mu_m_Mwl']), float(parameters['mu_z_Mwl'])]
        params_mWL_sigma = [float(parameters['sigma_Mwl_gal']), float(parameters['sigma_Mwl_int'])]
        rho_obs_mWL = float(parameters['rho'])
        which_mass_richness_rel = config_new['cluster_catalogue.mass_observable_relation']['which_relation']

        add_photoz = True if config_new['cluster_catalogue']['add_photometric_redshift']=='True' else False
        photoz_params = float(config_new['cluster_catalogue.photometric_redshift']['sigma_z0'])
    
        self.params_observable_mean = params_observable_mean
        self.params_observable_sigma = params_observable_sigma
        self.params_mWL_mean = params_mWL_mean
        self.params_mWL_sigma = params_mWL_sigma
        self.rho_obs_mWL = rho_obs_mWL
        self.add_photoz = add_photoz
        self.photoz_params = photoz_params
        self.which_mass_richness_rel = which_mass_richness_rel

    def mean_obs_power_law_f(self, log10M, z, params_observable_mean):
        log10m0, z0, observable_mu0, observable_muz, observable_mulog10m = params_observable_mean
        # Ensure all are float
        log10m0 = float(log10m0)
        z0 = float(z0)
        observable_mu0 = float(observable_mu0)
        observable_muz = float(observable_muz)
        observable_mulog10m = float(observable_mulog10m)

        # Ensure z and log10M are numeric arrays
        z = np.array(z, dtype=float)
        log10M = np.array(log10M, dtype=float)

        observable_mu = observable_mu0 + observable_muz * np.log10((1+z)/(1 + z0)) + observable_mulog10m * (log10M-log10m0)
        return observable_mu

    def sigma_obs_power_law_f(self, log10M, z, params_observable_sigma):
        log10m0, z0, observable_sigma0, observable_sigmaz, observable_sigmalog10m = params_observable_sigma
        observable_sigma = observable_sigma0 + observable_sigmaz * np.log10((1+z)/(1 + z0)) + observable_sigmalog10m * (log10M-log10m0)
        return observable_sigma

    def mean_log10mWL_f(self, log10M, z, params_mWL_mean):

        mu_0_Mwl, mu_m_Mwl, mu_z_Mwl = params_mWL_mean
        # Need pivot points from observable mean parameters
        log10m0 = self.params_observable_mean[0]
        z0 = self.params_observable_mean[1]
        return mu_0_Mwl + mu_m_Mwl * (log10M - log10m0) + mu_z_Mwl * np.log10((1+z)/(1 + z0))
    
    def sigma_log10mWL_f(self, log10M, z, params_mWL_sigma):

        sigma_mWLgal, sigma_mWLint = params_mWL_sigma
        sigma2 = sigma_mWLgal**2 + sigma_mWLint**2
        return sigma2**.5*np.ones(len(log10M))

    def generate_observables_from_halo(self, log10M, z):

        mean_lnobs = self.mean_obs_power_law_f(log10M, z, self.params_observable_mean)
        sigma_lnobs = self.sigma_obs_power_law_f(log10M, z, self.params_observable_sigma)
        
        if self.which_mass_richness_rel=='Gauss+Poiss-corr':
            mean_lnobs = mean_lnobs - 0.5 * np.exp(-mean_lnobs+0.5*sigma_lnobs**2) - (1/12)*np.exp(-2*mean_lnobs+2*sigma_lnobs**2)
        sigma_lnobs2 = sigma_lnobs**2
        #add poisson noise
        sigma_lnobs2 = sigma_lnobs2 + (np.exp(mean_lnobs)-1)/np.exp(2*mean_lnobs)
        sigma_lnobs = sigma_lnobs2**.5
        
        mean_log10mWL = self.mean_log10mWL_f(log10M, z, self.params_mWL_mean)
        sigma_log10mWL = self.sigma_log10mWL_f(log10M, z, self.params_mWL_sigma)
        
        rho = self.rho_obs_mWL

        lnobs_noise = np.random.normal(loc=0, scale=sigma_lnobs)
        lnobs = mean_lnobs + lnobs_noise

        cond_mean_log10mWL = mean_log10mWL + rho * (sigma_log10mWL / sigma_lnobs) * (lnobs - mean_lnobs)
        cond_sigma_log10mWL = sigma_log10mWL * np.sqrt(1 - rho**2)
        log10Mwl = cond_mean_log10mWL + np.random.normal(loc=0, scale=cond_sigma_log10mWL)

        if self.add_photoz:
            z = photometric_redshift(z, self.photoz_params)
        
        return np.exp(lnobs), log10Mwl, z

    
    