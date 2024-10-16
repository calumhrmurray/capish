import numpy as np
import pyccl as ccl
import numpy as np
import scipy
from scipy import stats
from scipy.integrate import quad,simps, dblquad
from scipy import interpolate
import sys

class ClusterAbundance():
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
    def ___init___(self):
        self.name = 'Cosmological prediction for cluster abundance cosmology'
        
    def set_cosmology(self, cosmo = 1, massdef = None, hmd = None):
        r"""
        Attributes:
        ----------
        cosmo : CCL cosmology object
        mass_def: CCL object
            mass definition object of CCL
        hmd: CCL object
            halo mass distribution object from CCL
        """
        self.cosmo = cosmo
        self.massdef = massdef
        self.hmd = hmd
        #self.massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
        #self.hmd = ccl.halos.hmfunc.MassFuncDespali16(self.cosmo, mass_def=self.massdef)
        
    def dndlog10M(self, log10M, z):
        r"""
        Attributes:
        -----------
        log10M : array
            \log_{10}(M), M dark matter halo mass
        z : float
            halo redshift
        Returns:
        --------
        hmf : array
            halo mass function for the corresponding masses and redshift
        """
        hmf = self.hmd.__call__(self.cosmo, 10**np.array(log10M), 1./(1. + z))
        return hmf

    def dVdzdOmega(self,z):
        r"""
        Attributes:
        ----------
        z : float
            redshift
        Returns:
        -------
        dVdzdOmega_value : float
            differential comoving volume 
        """
        a = 1./(1. + z)
        da = ccl.background.angular_diameter_distance(self.cosmo, a)
        ez = ccl.background.h_over_h0(self.cosmo, a) 
        dh = ccl.physical_constants.CLIGHT_HMPC / self.cosmo['h']
        dVdzdOmega_value = dh * da * da/( ez * a ** 2)
        return dVdzdOmega_value

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
            grid[:,i] = self.dndlog10M(self.logm_grid ,z) * self.dVdzdOmega(z)
        self.dN_dzdlogMdOmega = grid
        #self.dNdzdlogMdOmega_interpolation = interpolate.interp2d(self.z_grid, 
        #                                                        self.logm_grid, 
        ##                                                        self.dN_dzdlogMdOmega, 
        #                                                        kind='cubic')
        
    def compute_halo_bias_grid_MZ(self, z_grid = 1, logm_grid = 1, halobiais = 1):
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
        self.halo_bias_model = halobiais
        for i, z in enumerate(self.z_grid):
            hb = self.halo_bias_model.__call__(self.cosmo, 10**self.logm_grid, 1./(1. + z), )#mdef_other = self.massdef)
            grid[:,i] = hb
        self.halo_biais = grid
        #self.halo_biais_interpolation = interpolate.interp2d(self.z_grid, 
                   #                                             self.logm_grid, 
                   #                                             self.halo_biais, 
                   #                                             kind='cubic')
        
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
        
        if method == 'exact_CCL':
            def __integrand__(logm, z):
                a = self.sky_area * self.dVdzdOmega(z) * self.dndlog10M(logm, z)  
                b = self.halo_bias_model.get_halo_bias(self.cosmo, 10**logm, 1./(1. + z), mdef_other = self.massdef)
                return a*b
            for i, proxy_bin in enumerate(Proxy_bin):
                for j, z_bin in enumerate(Redshift_bin):
                    halo_biais_matrix[j,i] = scipy.integrate.dblquad(__integrand__, 
                                                               z_bin[0], z_bin[1], 
                                                               lambda x: proxy_bin[0], 
                                                               lambda x: proxy_bin[1])[0]
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
                    
        if method == 'bin_format':
            
            return 0
                    
        if method == 'exact_CCL':
            def dN_dzdlogMdOmega(logm, z):
                return self.sky_area * self.dVdzdOmega(z) * self.dndlog10M(logm, z)
            for i, proxy_bin in enumerate(Proxy_bin):
                for j, z_bin in enumerate(Redshift_bin):
                    N_th_matrix[j,i] = scipy.integrate.dblquad(dN_dzdlogMdOmega, 
                                                               z_bin[0], z_bin[1], 
                                                               lambda x: proxy_bin[0], 
                                                               lambda x: proxy_bin[1])[0]

        return N_th_matrix
    
    def multiplicity_function_individual_MZ(self, z = .1, logm = 14, method = 'interp'):
        r"""
        Attributes:
        -----------
        z: array
            list of redshifs
        logm: array
            list of dark matter halo masses
        method: str
            method to use to compute multiplicity function
            "interp": use interpolated multiplicity function
            "exact_CCL": idividual CCL prediction
        Returns:
        --------
        dN_dzdlogMdOmega : array
            multiplicity function for the corresponding redshifts and masses
        """
        if method == 'interp':
            dN_dzdlogMdOmega_fct = interpolate.RectBivariateSpline(self.logm_grid, self.z_grid, 
                                                                   self.dN_dzdlogMdOmega)
            dN_dzdlogMdOmega = dN_dzdlogMdOmega_fct(logm, z, grid = False)    
        if method == 'exact_CCL':
            dN_dzdlogMdOmega = np.zeros(len(z))
            for i, z_ind, logm_ind in zip(np.arange(len(z)), z, logm):
                dN_dzdlogMdOmega[i] = self.dndlog10M(logm_ind, z_ind) * self.dVdzdOmega(z_ind)
        return dN_dzdlogMdOmega

