from scipy.stats import norm, poisson
import numpy as np

class Richness_mass_relation():

    def ___init___(self,):
        self.name = 'mass-richness relation'
    
    def select(self, which = 'log_normal'):
        self.which = which

    # def proxy_mu_f(self, logm, z, theta_rm):
    #     r"""proxy mu"""
    #     log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
    #     proxy_mu = proxy_mu0 + proxy_muz * np.log((1+z)/(1 + z0)) + proxy_mulog10m * (logm-log10m0)
    #     return proxy_mu

    # def proxy_sigma_f(self, logm, z, theta_rm):
    #     r"""proxy sigma, intrinsic"""
    #     log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
    #     proxy_sigma = proxy_sigma0 + proxy_sigmaz * np.log((1+z)/(1 + z0)) + proxy_sigmalog10m * (logm-log10m0)
    #     return proxy_sigma

    def proxy_mu_f(self, logm, z, theta_rm):
        r"""proxy mu"""
        M_min, alpha_lambda, beta_lambda, gamma_lambda, sigma_lambda = theta_rm
        proxy_mu = alpha_lambda + beta_lambda * np.log10(10**logm - M_min) + gamma_lambda * np.log10(1+z)
        return proxy_mu * np.log(10)

    def proxy_sigma_f(self, logm, z, theta_rm):
        r"""proxy sigma, intrinsic"""
        M_min, alpha_lambda, beta_lambda, gamma_lambda, sigma_lambda = theta_rm
        proxy_sigma = sigma_lambda
        return proxy_sigma


    def pdf_richness_mass_relation(self, richness, logm, z, theta_rm,):
        r"""
        Attributes:
        -----------
        richness : array
            cluster richness
        logm: array
            logm of halo mass
        z : float
            cluster redshift
        theta_rm: array
            parameters of purity
        Returns:
        --------
        rm : array
            richness-mass relation P(lambda|m,z)
            This is only necessary for the likelihood implementation
        """
        proxy_mu = self.proxy_mu_f(logm, z, theta_rm)
        proxy_sigma = self.proxy_sigma_f(logm, z, theta_rm)
        proxy_sigma2 = proxy_sigma**2 + (np.exp(proxy_mu)-1)/np.exp(2*proxy_mu)  
        proxy_sigma = proxy_sigma2**.5

        return (1/richness)*np.exp(-(np.log(richness)-proxy_mu)**2/(2*proxy_sigma**2))/np.sqrt(2*np.pi*proxy_sigma**2)

