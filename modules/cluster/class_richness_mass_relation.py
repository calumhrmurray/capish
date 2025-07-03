from scipy.stats import norm, poisson
import numpy as np

class Richness_mass_relation():

    def ___init___(self,):
        self.name = 'mass-richness relation'
    
    def select(self, which = 'log_normal'):
        self.which = which

    def proxy_mu_f(self, logm, z, theta_rm, which_model = 'constantins model'):
        r"""proxy mu"""
        if which_model == 'constantins model':
            #m0, z0, lnlambda0, muz, mum, sigma_lnlambda0, sigma_z, sigma_m
            log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
            proxy_mu = proxy_mu0 + proxy_muz * np.log((1+z)/(1 + z0)) + proxy_mulog10m * (logm-log10m0)
            return proxy_mu
        if which_model == 'power_law':
            #z0, lnlambda0, muz, mum
            z0, c_l, beta_l, alpha_l, cosmo = theta_rm
            mean_ln_l = c_l + beta_l * np.log( cosmo.h_over_h0(1/(1+z)) / cosmo.h_over_h0(1/(1 + self.z_p) ) ) + alpha_l * mu

    def proxy_sigma_f(self, logm, z, theta_rm):
        r"""proxy sigma, intrinsic"""
        log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
        proxy_sigma = proxy_sigma0 + proxy_sigmaz * np.log((1+z)/(1 + z0)) + proxy_sigmalog10m * (logm-log10m0)
        return proxy_sigma

    def lnLambda_random(self, logm, z, theta_rm):

        r"""
        Attributes:
        -----------
        logm: array
            logm of halo mass
        z : float
            cluster redshift
        theta_rm: array
            parameters of purity
        Returns:
        --------
        rm : array
            sample of richness
        """

        mu = self.proxy_mu_f(logm, z, theta_rm)
        sigma = self.proxy_sigma_f(logm, z, theta_rm)
        
        if 'poisson_log_scatter' in self.which: 
            sigma2 = sigma**2 + (np.exp(mu)-1)/np.exp(2*mu)
            sigma = sigma2**.5
            return mu + sigma * np.random.randn(len(logm))
        
        if 'poisson_scatter' in self.which: 
            mu_rand = mu + sigma * np.random.randn(len(logm))
            lam = poisson.rvs(np.exp(mu_rand))
            return np.log(lam)
        
        if 'poisson_scatter' not in self.which and 'poisson_log_scatter' not in self.which: 
            return mu + sigma * np.random.randn(len(logm))

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

        if 'poisson_scatter' in self.which or 'poisson_log_scatter' in self.which: 
            proxy_sigma2 = proxy_sigma**2 + (np.exp(proxy_mu)-1)/np.exp(2*proxy_mu)  
            proxy_sigma = proxy_sigma2**.5

        return (1/richness)*np.exp(-(np.log(richness)-proxy_mu)**2/(2*proxy_sigma**2))/np.sqrt(2*np.pi*proxy_sigma**2)

    
    