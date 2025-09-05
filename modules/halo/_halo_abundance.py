import numpy as np
import pyccl as ccl
import numpy as np
import scipy
from scipy import stats
from scipy.integrate import quad, dblquad
# Handle scipy version compatibility
try:
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps
from scipy import interpolate
import sys

class HaloAbundance():
    r"""
        1. computation of the cosmological prediction for cluster abundance cosmology, for 
            a. cluster count in mass and redhsift intervals (binned approach)
            b. cluster count with individual masses and redshifts (un-binned approach)
            c. cluster count in mass proxy and redhsift intervals (binned approach)
            d. cluster count with individual mass proxies and redshifts (un-binned approach)
        Core Cosmology Library (arXiv:1812.05995) as backend for:
        1. comoving differential volume
        2. halo mass function
    """
    def __init__(self, CCLCosmologyObject = None, CCLHmf = None, CCLBias = None, sky_area = None):
        self.name = 'Cosmological prediction for cluster abundance cosmology'
        self.CCLCosmologyObject = CCLCosmologyObject
        self.CCLHmf = CCLHmf
        self.CCLBias = CCLBias
        self.sky_area = sky_area
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
        return self.CCLHmf.__call__(cosmo, 10**np.array(logm), 1./(1. + z))
    
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

    def compute_theoretical_Sij(self, Z_bin, cosmo, f_sky, S_ij_type = 'full_sky_rescaled_approx', path_to_mask = None):
        
        default_cosmo_params = {'omega_b':cosmo['Omega_b']*cosmo['h']**2, 
                                'omega_cdm':cosmo['Omega_c']*cosmo['h']**2, 
                                'H0':cosmo['h']*100, 
                                'n_s':cosmo['n_s'], 
                                'sigma8': cosmo['sigma8'],
                                'output' : 'mPk'}
        
        # this should be in a settings file somewhere
        z_arr = np.linspace(0.1,1.2,1000)
        nbins_T   = len(Z_bin)
        windows_T = np.zeros((nbins_T,len(z_arr)))
        
        for i, z_bin in enumerate(Z_bin):
                Dz = z_bin[1]-z_bin[0]
                z_arr_cut = z_arr[(z_arr > z_bin[0])*(z_arr < z_bin[1])]
                for k, z in enumerate(z_arr):
                    if ((z>z_bin[0]) and (z<=z_bin[1])):
                        windows_T[i,k] = 1
        
        if S_ij_type == 'full_sky_rescaled_approx':  
            Sij_fullsky = pyssc.Sij_fullsky(z_arr, windows_T, order=1, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0)
            Sij_partialsky = Sij_fullsky/f_sky
            
        elif S_ij_type == 'full_sky_rescaled': 
            Sij_fullsky = pyssc.Sij(z_arr, windows_T, order=1, sky='full', method='classic', 
                                    cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0,
                                    precision=10, clmask=None, mask=None, mask2=None, 
                                    var_tol=0.05, machinefile=None, Nn=None, Np='default', 
                                    AngPow_path=None, verbose=False, debug=False)
            Sij_partialsky = Sij_fullsky/f_sky
        
        elif S_ij_type == 'exact':
            Sij_partialsky = pyssc.Sij(z_arr, windows_T, order=1, sky='psky', method='classic', 
                                    cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0,
                                    precision=10, clmask=None, mask=path_to_mask, mask2=None, 
                                    var_tol=0.05, machinefile=None, Nn=None, Np='default', 
                                    AngPow_path=None, verbose=False, debug=False)
        return Sij_partialsky 

    def compute_theoretical_sigma2ij_fullsky(self, cosmo_ccl, z_grid):
        import PySSC
        
        default_cosmo_params = {'omega_b': cosmo_ccl['Omega_b']*cosmo_ccl['h']**2, 
                                'omega_cdm': cosmo_ccl['Omega_c']*cosmo_ccl['h']**2, 
                                'H0': cosmo_ccl['h']*100, 
                                'n_s': cosmo_ccl['n_s'], 
                                'sigma8': cosmo_ccl['sigma8'],
                                'output' : 'mPk'}
        
        return PySSC.sigma2_fullsky(z_grid, cosmo_params=default_cosmo_params, cosmo_Class=None)

    
    def compute_multiplicity_grid_MZ(self, z_grid = 1, logm_grid = 1):
        r"""
        Attributes:
        -----------
        z_grid : array
            redshift grid
        logm_grid : array
            logm grid
        Returns:
        --------
        dN_dzdlogMdOmega : array
            tabulated multiplicity function over the redshift and logmass grid
        dzdlogMdOmega_interpolation : function
            interpolated function over the tabulated multiplicity grid
        """
        self.z_grid = z_grid
        self.logm_grid = logm_grid
        grid = np.zeros([len(self.logm_grid), len(self.z_grid)])
        for i, z in enumerate(self.z_grid):
            grid[:,i] = self.dndlog10M(self.logm_grid ,z, self.CCLCosmologyObject) * self.dVdzdOmega(z, self.CCLCosmologyObject)
        self.dN_dzdlogMdOmega = grid
        
    def compute_halo_bias_grid_MZ(self, z_grid = 1, logm_grid = 1):
        r"""
        Attributes:
        -----------
        z_grid : array
            redshift grid
        logm_grid : array
            logm grid
        Returns:
        --------
        dN_dzdlogMdOmega : array
            tabulated multiplicity function over the redshift and logmass grid
        dzdlogMdOmega_interpolation : function
            interpolated function over the tabulated multiplicity grid
        """
        grid = np.zeros([len(self.logm_grid), len(self.z_grid)])
        for i, z in enumerate(self.z_grid):
            hb = self.CCLBias.__call__(self.CCLCosmologyObject, 10**self.logm_grid, 1./(1. + z))
            grid[:,i] = hb
        self.halo_biais = grid
        
    def Nhalo_bias_MZ(self, Redshift_bin = [], Proxy_bin = [], method = 'simps'): 
        r"""
        returns the predicted number count in mass-redshift bins
        Attributes:
        -----------
        Redshift_bin : list of lists
            list of redshift bins
        Proxy_bin : list of lists
            list of mass bins
        method : str
            method to be used for the cluster abundance prediction
            "simps": use simpson integral of the tabulated multiplicity
            "exact_CCL": use scipy.dblquad to integer CCL multiplicity function
        Returns:
        --------
        N_th_matrix: ndarray
            matrix for the cluster abundance prediction in redshift and mass bins
        """
        halo_biais_matrix = np.zeros([len(Redshift_bin), len(Proxy_bin)]) 
        if method == 'simps':               
            index_proxy = np.arange(len(self.logm_grid))
            index_z = np.arange(len(self.z_grid))
            for i, proxy_bin in enumerate(Proxy_bin):
                mask_proxy = (self.logm_grid >= proxy_bin[0])*(self.logm_grid <= proxy_bin[1])
                proxy_cut = self.logm_grid[mask_proxy]
                index_proxy_cut = index_proxy[mask_proxy]
                proxy_cut[0], proxy_cut[-1] = proxy_bin[0], proxy_bin[1]
                for j, z_bin in enumerate(Redshift_bin):
                    z_down, z_up = z_bin[0], z_bin[1]
                    mask_z = (self.z_grid >= z_bin[0])*(self.z_grid <= z_bin[1])
                    z_cut = self.z_grid[mask_z]
                    index_z_cut = index_z[mask_z]
                    z_cut[0], z_cut[-1] = z_down, z_up
                    integrand = self.sky_area * np.array([self.dN_dzdlogMdOmega[:,k][mask_proxy] * self.halo_biais[:,k][mask_proxy] for k in index_z_cut])
                    halo_biais_matrix[j,i] = simps(simps(integrand, proxy_cut), z_cut)
            return halo_biais_matrix

    def Cluster_Abundance_MZ(self, Redshift_bin = [], Proxy_bin = [], method = 'dblquad_interp'): 
        r"""
        returns the predicted number count in mass-redshift bins
        Attributes:
        -----------
        Redshift_bin : list of lists
            list of redshift bins
        Proxy_bin : list of lists
            list of mass bins
        method : str
            method to be used for the cluster abundance prediction
            "simps": use simpson integral of the tabulated multiplicity
            "dblquad_interp": integer interpolated multiplicity function
            "exact_CCL": use scipy.dblquad to integer CCL multiplicity function
        Returns:
        --------
        N_th_matrix: ndarray
            matrix for the cluster abundance prediction in redshift and mass bins
        """
        N_th_matrix = np.zeros([len(Redshift_bin), len(Proxy_bin)])
        if method == 'dblquad_interp':
            for i, proxy_bin in enumerate(Proxy_bin):
                for j, z_bin in enumerate(Redshift_bin):
                    N_th_matrix[j,i] = self.sky_area * dblquad(self.dNdzdlogMdOmega_interpolation, 
                                                   proxy_bin[0], proxy_bin[1], 
                                                   lambda x: z_bin[0], 
                                                   lambda x: z_bin[1])[0]
                    
        if method == 'simps':
            index_proxy = np.arange(len(self.logm_grid))
            index_z = np.arange(len(self.z_grid))
            for i, proxy_bin in enumerate(Proxy_bin):
                mask_proxy = (self.logm_grid >= proxy_bin[0])*(self.logm_grid <= proxy_bin[1])
                proxy_cut = self.logm_grid[mask_proxy]
                index_proxy_cut = index_proxy[mask_proxy]
                proxy_cut[0], proxy_cut[-1] = proxy_bin[0], proxy_bin[1]
                for j, z_bin in enumerate(Redshift_bin):
                    z_down, z_up = z_bin[0], z_bin[1]
                    mask_z = (self.z_grid >= z_bin[0])*(self.z_grid <= z_bin[1])
                    z_cut = self.z_grid[mask_z]
                    index_z_cut = index_z[mask_z]
                    z_cut[0], z_cut[-1] = z_down, z_up
                    integrand = np.array([self.dN_dzdlogMdOmega[:,k][mask_proxy] for k in index_z_cut])
                    N_th = self.sky_area * simps(simps(integrand, proxy_cut), z_cut)
                    N_th_matrix[j,i] = N_th

        return N_th_matrix

    def Cluster_NMass_MZ(self, Redshift_bin = [], Proxy_bin = [], method = 'dblquad_interp', power = 1): 
        r"""
        returns the predicted number count in mass-redshift bins
        Attributes:
        -----------
        Redshift_bin : list of lists
            list of redshift bins
        Proxy_bin : list of lists
            list of mass bins
        method : str
            method to be used for the cluster abundance prediction
            "simps": use simpson integral of the tabulated multiplicity
            "dblquad_interp": integer interpolated multiplicity function
            "exact_CCL": use scipy.dblquad to integer CCL multiplicity function
        Returns:
        --------
        N_th_matrix: ndarray
            matrix for the cluster abundance prediction in redshift and mass bins
        """
        N_th_matrix = np.zeros([len(Redshift_bin), len(Proxy_bin)])
                    
        index_proxy = np.arange(len(self.logm_grid))
        index_z = np.arange(len(self.z_grid))
        for i, proxy_bin in enumerate(Proxy_bin):
            mask_proxy = (self.logm_grid >= proxy_bin[0])*(self.logm_grid <= proxy_bin[1])
            proxy_cut = self.logm_grid[mask_proxy]
            index_proxy_cut = index_proxy[mask_proxy]
            proxy_cut[0], proxy_cut[-1] = proxy_bin[0], proxy_bin[1]
            for j, z_bin in enumerate(Redshift_bin):
                z_down, z_up = z_bin[0], z_bin[1]
                mask_z = (self.z_grid >= z_bin[0])*(self.z_grid <= z_bin[1])
                z_cut = self.z_grid[mask_z]
                index_z_cut = index_z[mask_z]
                z_cut[0], z_cut[-1] = z_down, z_up
                integrand = np.array([self.dN_dzdlogMdOmega[:,k][mask_proxy] for k in index_z_cut])
                N_th = self.sky_area * simps(simps(integrand * 10 ** (proxy_cut * power), proxy_cut), z_cut)
                N_th_matrix[j,i] = N_th

        return N_th_matrix