from scipy.stats import norm, poisson
import numpy as np

class Richness_mass_relation():

    def ___init___(self,):
        self.name = 'mass-richness relation'
    
    def select(self, which = 'log_normal'):
        self.which = which

    def proxy_mu_f(self, logm, z, theta_rm, which_model = 'constantins model'):
        r"""proxy mu"""
        log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
        proxy_mu = proxy_mu0 + proxy_muz * np.log((1+z)/(1 + z0)) + proxy_mulog10m * (logm-log10m0)
        return proxy_mu

    def proxy_sigma_f(self, logm, z, theta_rm):
        r"""proxy sigma, intrinsic"""
        log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
        proxy_sigma = proxy_sigma0 + proxy_sigmaz * np.log((1+z)/(1 + z0)) + proxy_sigmalog10m * (logm-log10m0)
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

        log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
        proxy_mu = self.proxy_mu_f(logm, z, theta_rm)
        proxy_sigma = self.proxy_sigma_f(logm, z, theta_rm)
        if self.which == 'Gauss+Poiss-corr':
           proxy_mu = proxy_mu - 0.5 * np.exp(-proxy_mu+0.5*proxy_sigma**2) - (1/12)*np.exp(-2*proxy_mu+2*proxy_sigma**2)
        proxy_sigma2 = proxy_sigma**2 + (np.exp(proxy_mu)-1)/np.exp(2*proxy_mu)  
        proxy_sigma = proxy_sigma2**.5

        return (1/richness)*np.exp(-(np.log(richness)-proxy_mu)**2/(2*proxy_sigma**2))/np.sqrt(2*np.pi*proxy_sigma**2)

