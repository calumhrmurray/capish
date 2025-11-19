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
        a_nc, b_nc, a_mc, b_mc = 2.5, 0, 13.5, 0
    else: a_nc, b_nc, a_mc, b_mc = params
    
    nc = a_nc + b_nc*(1+z)
    log10mc = a_mc + b_mc*(1+z)

    ratio = 10**log10m / (10**log10mc)
    
    return ratio ** nc /(1 + ratio ** nc)