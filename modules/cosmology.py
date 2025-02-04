from scipy.stats import norm
import numpy as np
import pyccl as ccl

class Cosmology():
    r"""
        Core Cosmology Library (arXiv:1812.05995) as backend for:
        1. comoving differential volume
        2. halo mass function
    """
    def __init__(self, hmf=None, bias_model=None):
        self.name = 'Cosmological prediction for cluster abundance cosmology'
        self.hmf = hmf
        self.bias_model = bias_model
        return None

    def dndlog10M(self, logm, z, cosmo):
        r"""
        Attributes:
        -----------
        log10M : array
            \log_{10}(M), M dark matter halo mass
        z : float
            halo redshift
        cosmo: CCL cosmology object
            cosmological parameters
        hmd: CCL hmd object
            halo definition
        Returns:
        --------
        hmf : array
            halo mass function for the corresponding masses and redshift
        """
        return self.hmf.__call__(cosmo, 10**np.array(logm), 1./(1. + z))
    
    def dVdzdOmega(self, z, cosmo):
        r"""
        Attributes:
        ----------
        z : float
            redshift
        cosmo: CCL cosmology
            cosmological parameters
        Returns:
        -------
        dVdzdOmega_value : float
            differential comoving volume 
        """
        a = 1./(1. + z)
        da = ccl.background.angular_diameter_distance(cosmo, a)
        ez = ccl.background.h_over_h0(cosmo, a) 
        dh = ccl.physical_constants.CLIGHT_HMPC / cosmo['h']
        dVdzdOmega_value = dh * da * da/( ez * a ** 2)
        return dVdzdOmega_value



