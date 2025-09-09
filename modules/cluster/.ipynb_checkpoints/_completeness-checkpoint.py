import numpy as np

def completeness(log10m, z, params = None):
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
    if params==None:
        a_nc, b_nc, a_mc, b_mc = 1.1321, 0.7751, 13.31, 0.2025
    else: a_nc, b_nc, a_mc, b_mc = params
    nc = a_nc + b_nc*(1+z)
    log10mc = a_mc + b_mc*(1+z)
    return np.exp(nc*np.log(10)*(log10m-log10mc))/(1+np.exp(nc*np.log(10)*(log10m-log10mc)))
