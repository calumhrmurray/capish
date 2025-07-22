from scipy.stats import norm
import numpy as np

def completeness_Aguena(logm, z, theta_completeness):
    r"""
    Attributes:
    -----------
    log10M : array
        \log_{10}(M), M dark matter halo mass
    z : float
        halo redshift
    theta_completeness: array
        parameters of completeness
    Returns:
    --------
    completeness : array
        completeness of cluster detection
    """
    logm0, c0, c1, nc = theta_completeness
    logm_scale = logm0 + c0 + c1*(1 + z)
    m_rescale = 10**logm/(10**logm_scale)
    return m_rescale**nc/(m_rescale**nc+1)

def completeness(log10m, z, theta_completeness):
    r"""
    Attributes:
    -----------
    log10M : array
        \log_{10}(M), M dark matter halo mass
    z : float
        halo redshift
    theta_completeness: array
        parameters of completeness
    Returns:
    --------
    completeness : array
        completeness of cluster detection
    """
    a_nc, b_nc, a_mc, b_mc = theta_completeness
    nc = a_nc + b_nc*(1+z)
    log10mc = a_mc + b_mc*(1+z)
    return np.exp(nc*np.log(10)*(log10m-log10mc))/(1+np.exp(nc*np.log(10)*(log10m-log10mc)))