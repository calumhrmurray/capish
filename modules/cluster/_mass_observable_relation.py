from scipy.stats import norm, poisson
import numpy as np


class Mass_observable_relation:
    def __init__(self, params_observable_mean, params_observable_stdd,
                       params_mWL_mean, params_mWL_stdd, 
                       rho, which = 'Gauss-only'):
    
        self.params_observable_mean = params_observable_mean
        self.params_observable_stdd = params_observable_stdd
        self.params_mWL_mean = params_mWL_mean
        self.params_mWL_stdd = params_mWL_stdd
        self.rho_obs_mWL = rho
        self.which = which

    def mean_obs_power_law_f(self, logm, z, params_observable_mean):
        log10m0, z0, observable_mu0, observable_muz, observable_mulog10m = params_observable_mean
        observable_mu = observable_mu0 + observable_muz * np.log((1+z)/(1 + z0)) + observable_mulog10m * (logm-log10m0)
        return observable_mu

    def stdd_obs_power_law_f(self, logm, z, params_observable_stdd):
        log10m0, z0, observable_sigma0, observable_sigmaz, observable_sigmalog10m = params_observable_stdd
        observable_sigma = observable_sigma0 + observable_sigmaz * np.log((1+z)/(1 + z0)) + observable_sigmalog10m * (logm-log10m0)
        return observable_sigma

    def mean_log10mWL_f(self, logm, z, params_mWL_mean):

        a_WL, b_WL = params_mWL_mean
        return a_WL * logm + b_WL
    
    def stdd_log10mWL_f(self, logm, z, params_mWL_stdd):

        stdd_mWLgal, stdd_mWLint = params_mWL_stdd
        stdd2 = stdd_mWLgal**2 + stdd_mWLint**2
        return stdd2**.5*np.ones(len(logm))

    def generate_mWL_richness(self, logm, z):

        mean_lnobs = self.mean_obs_power_law_f(logm, z, self.params_observable_mean)
        stdd_lnobs = self.stdd_obs_power_law_f(logm, z, self.params_observable_stdd)
        
        if self.which == 'Gauss+Poiss-corr':
            mean_lnobs = mean_lnobs - 0.5 * np.exp(-mean_lnobs+0.5*stdd_lnobs**2) - (1/12)*np.exp(-2*mean_lnobs+2*stdd_lnobs**2)
        stdd_lnobs2 = stdd_lnobs**2
        stdd_lnobs2 = stdd_lnobs2 + (np.exp(mean_lnobs)-1)/np.exp(2*mean_lnobs)
        stdd_lnobs = stdd_lnobs2**.5
        
        mean_log10mWL = self.mean_log10mWL_f(logm, z, self.params_mWL_mean)
        stdd_log10mWL = self.stdd_log10mWL_f(logm, z, self.params_mWL_stdd)
        
        rho = self.rho_obs_mWL

        lnobs_noise = np.random.normal(loc=0, scale=stdd_lnobs)
        lnobs = mean_lnobs + lnobs_noise

        cond_mean_log10mWL = mean_log10mWL + rho * (stdd_log10mWL / stdd_lnobs) * (lnobs - mean_lnobs)
        cond_stdd_log10mWL = stdd_log10mWL * np.sqrt(1 - rho**2)
        log10Mwl = cond_mean_log10mWL + np.random.normal(loc=0, scale=cond_stdd_log10mWL)
        
        return np.exp(lnobs), log10Mwl, z

    
    