import numpy as np

def purity(richness, z, params = None):
    r"""
    Attributes:
    -----------
    richness : array
        cluster richness
    z : float
        cluster redshift
    theta_purity: array
        parameters of purity
    Returns:
    --------
    purity : array
        purity of cluster detection
    """
    if params==None:
        a_nc, b_nc, a_rc, b_rc = 1.98, 0.81, 2.21, -0.65
        theta_purity = [a_nc, b_nc, a_rc, b_rc]
    else: 
        a_nc, b_nc, a_rc, b_rc = params
    nc = a_nc + b_nc * (1 + z)
    lnrc = a_rc + b_rc * (1 + z)
    lnr = np.log(richness)
    lnr_rescaled = lnr/lnrc
  
    return (lnr_rescaled)**nc / ((lnr_rescaled)**nc + 1)
