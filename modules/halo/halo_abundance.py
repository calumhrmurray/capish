import numpy as np
import pyccl as ccl
import numpy as np
import scipy
from scipy import stats
from scipy.integrate import quad,simps, dblquad
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
    def __init__(self, CosmologyObject = None , sky_area = None ):
        self.name = 'Cosmological prediction for cluster abundance cosmology'
        self.cosmo_prediction = CosmologyObject
        self.sky_area = sky_area
        return None
        
    def set_cosmology(self, cosmo = None):
        r"""
        Attributes:
        ----------
        cosmo : CCL cosmology object
        mass_def: CCL object
            mass definition object of CCL
        hmf: CCL object
            halo mass distribution object from CCL
        """
        self.cosmo = cosmo

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
            grid[:,i] = self.cosmo_prediction.dndlog10M(self.logm_grid ,z, self.cosmo) * self.cosmo_prediction.dVdzdOmega(z, self.cosmo)
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
            hb = self.cosmo_prediction.bias_model.__call__(self.cosmo, 10**self.logm_grid, 1./(1. + z))
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