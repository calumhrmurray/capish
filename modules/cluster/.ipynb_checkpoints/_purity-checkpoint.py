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
        a_nc, b_nc, a_rc, b_rc = np.log(10)*0.8612, np.log(10)*0.3527, 2.2183, -0.6592
        theta_purity = [a_nc, b_nc, a_rc, b_rc]
    else: 
        a_nc, b_nc, a_rc, b_rc = params
    nc = a_nc + b_nc * (1 + z)
    lnrc = a_rc + b_rc * (1 + z)
    lnr = np.log(richness)
    lnr_rescaled = lnr/lnrc
  
    return (lnr_rescaled)**nc / ((lnr_rescaled)**nc + 1)
