from scipy.stats import norm
import numpy as np

def proxy_mu_f(logm, z, theta_rm):
    r"""proxy mu"""
    log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
    proxy_mu = proxy_mu0 + proxy_muz * np.log((1+z)/(1 + z0)) + proxy_mulog10m * (logm-log10m0)
    return proxy_mu

def proxy_sigma_f(logm, z, theta_rm):
    r"""proxy sigma"""
    log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
    proxy_sigma = proxy_sigma0 + proxy_sigmaz * np.log((1+z)/(1 + z0)) + proxy_sigmalog10m * (logm-log10m0)
    return proxy_sigma

def pdf_richness_mass_relation(richness, logm, z, theta_rm):
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
    """
    log10m0, z0, proxy_mu0, proxy_muz, proxy_mulog10m, proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m = theta_rm
    proxy_mu = proxy_mu_f(logm, z, theta_rm)
    proxy_sigma = proxy_sigma_f(logm, z, theta_rm)
    return (1/richness)*np.exp(-(np.log(richness)-proxy_mu)**2/(2*proxy_sigma**2))/np.sqrt(2*np.pi*proxy_sigma**2)

def lnLambda_random(logm, z, theta_rm):

    mu = proxy_mu_f(logm, z, theta_rm)
    sigma = proxy_sigma_f(logm, z, theta_rm)
    return mu + sigma * np.random.randn(len(logm))
