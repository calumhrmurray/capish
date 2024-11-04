import numpy as np
from itertools import combinations, chain
import healpy
import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/PySSC/')
import PySSC

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

class Covariance_matrix():
    r"""
    Class for the computation of covariance matrices for cluster abundance:
    a. Bootstrap covariance matrix
    b. Jackknife covariance matrix
    c. Sample covariance matrix
    """
    def __init__(self):
        self.name = 'name'
        return None

    def compute_boostrap_covariance(self, catalog = None, proxy_colname = 'M200c', 
                                    redshift_colname = 'redshift', proxy_corner = None, 
                                    z_corner = None, n_boot = 100, fct_modify = None):
        r"""
        Attributes:
        -----------
        catalog: Table
            single catalog of clusters (ra, dec, z, proxy, etc...)
        proxy_colname: str
            name of the proxy column
        redshift_colname: str
            name of the redshift column
        proxy_corner: str
            values of proxues to be binned
        z_corner: str
            values of redshifts to be binned
        n_boot: int
            number of bootstrap resampling
        fct_modify: fct
            function to add optional modifications
        Returns:
        --------
        cov_N: array
            bootstrap covariance matrix
        """
        proxy, redshift = catalog[proxy_colname], catalog[redshift_colname]
        index = np.arange(len(proxy))
        data_boot = []
        for i in range(n_boot):
            index_bootstrap = np.random.choice(index, len(index))
            data, proxy_edges, z_edges = np.histogram2d(redshift[index_bootstrap], 
                                                        proxy[index_bootstrap],
                                                   bins=[z_corner, proxy_corner])
            data_boot.append(data.flatten())
        data_boot = np.array(data_boot)
        N = np.stack((data_boot.astype(float)), axis = 1)
        mean = np.mean(data_boot, axis = 0)
        cov_N = np.cov(N, bias = False)
        self.Bootstrap_covariance_matrix = cov_N

    def compute_jackknife_covariance(self, catalog = None, proxy_colname = 'M200c', 
                                           redshift_colname = 'redshift',z_corner = None, 
                                           proxy_corner = None, ra_colname = None, 
                                           dec_colname = None, n_power = 32, N_delete = 1):
        r"""
        Attributes:
        -----------
        catalog: Table
            single catalog of clusters (ra, dec, z, proxy, etc...)
        proxy_colname: str
            name of the proxy column
        redshift_colname: str
            name of the redshift column
        proxy_corner: str
            values of proxues to be binned
        ra_colname: str
            name of the ra column
        dec_colname: str
            name of the dec column
        z_corner: str
            values of redshifts to be binned
        n_power: int
            defines the number of healpix pixels
        N_delete: int
            number of jackknife region to delete each repetition
        Returns:
        --------
        cov_N: array
            Jackknife covariance matrix
        """
        proxy, redshift = catalog[proxy_colname], catalog[redshift_colname]
        ra, dec =  catalog[ra_colname], catalog[dec_colname]
        index = np.arange(len(proxy))
        healpix = healpy.ang2pix(2**n_power, ra, dec, nest=True, lonlat=True)
        healpix_list_unique = np.unique(healpix)
        healpix_combination_delete = list(combinations(healpix_list_unique, N_delete))
        data_jack = []
        for i, hp_list_delete in enumerate(healpix_combination_delete):
                mask_in_area = np.isin(healpix, hp_list_delete)
                mask_out_area = np.invert(mask_in_area)
                data, mass_edges, z_edges = np.histogram2d(redshift[mask_out_area], 
                                                           proxy[mask_out_area],
                                                           bins=[z_corner, proxy_corner])     
                data_jack.append(data.flatten())
        data_jack = np.array(data_jack)
        N = np.stack((data_jack.astype(float)), axis = 1)
        n_jack = len(healpix_combination_delete)
        cov_N = (n_jack - 1) * np.cov(N, bias = False,ddof=0)
        coeff = (n_jack-N_delete)/(N_delete*n_jack)
        self.Jackknife_covariance_matrix = cov_N * coeff
        
    def compute_sample_covariance(self, proxy_colname = 'M200c', redshift_colname = 'redshift',
                                z_corner = None, proxy_corner = None, catalogs_name=None, 
                                        fct_open = None, fct_modify = None):
        r"""
        Attributes:
        -----------
        proxy_colname: str
            name of the proxy column
        redshift_colname: str
            name of the redshift column
        proxy_corner: str
            values of proxues to be binned
        z_corner: str
            values of redshifts to be binned
        fct_open: fct
            opens individual catalogs
       fct_modify: fct
            modifies individual catalog
        Returns:
        --------
        cov_N: array
            Sample covariance covariance matrix
        """
        data_list = []
        for cat_name in catalogs_name:
            catalog = fct_open(cat_name)
            if fct_modify != None: fct_modify(catalog)
            data_individual, proxy_edges, z_edges = np.histogram2d(catalog[redshift_colname],
                                                                   catalog[proxy_colname], 
                                                                  bins=[z_corner, proxy_corner])
            data_list.append(data_individual.flatten())
        data = np.array(data_list)
        self.data_all_catalog = data
        N = np.stack((data.astype(float)), axis = 1)
        mean = np.mean(N, axis = 1)
        cov_N = np.cov(N, bias = False)
        self.covariance_matrix = cov_N
        self.mu = mean
        return cov_N
    
#     def matter_fluctuation_amplitude_fullsky(self, Z_bin, cosmo = None, approx = False):
#         r"""
#         Attributes:
#         -----------
#         Redshift_bin: array
#             list of redshift bins
#         Returns:
#         --------
#         Sij: array
#             matter fluctuation amplitude in redshift bins
#         r"""
#         if approx == False:
#             z_arr = np.linspace(0.05,2.5,3000)
#             nbins_T   = len(Z_bin)
#             windows_T = np.zeros((nbins_T,len(z_arr)))
#             for i, z_bin in enumerate(Z_bin):
#                 Dz = z_bin[1]-z_bin[0]
#                 z_arr_cut = z_arr[(z_arr > z_bin[0])*(z_arr < z_bin[1])]
#                 for k, z in enumerate(z_arr):
#                     if ((z>z_bin[0]) and (z<=z_bin[1])):
#                         windows_T[i,k] = 1/Dz  
#             #use F. Lacasa code PySSC for computing Sij
#             Sij = PySSC.Sij(z_arr,windows_T)
#             return Sij
            
#         else:
            
#             default_cosmo_params = {'omega_b':cosmo['Omega_b']*cosmo['h']**2, 
#                                 'omega_cdm':cosmo['Omega_c']*cosmo['h']**2, 
#                                 'H0':cosmo['h']*100, 
#                                 'n_s':cosmo['n_s'], 
#                                 'sigma8': cosmo['sigma8'],
#                                 'output' : 'mPk'}
#             return PySSC.Sij_alt_fullsky(Z_bin, [None], order=2, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0)

    def matter_fluctuation_amplitude_fullsky(self, Z_bin, cosmo = None, approx = False):
        r"""
        Attributes:
        -----------
        Redshift_bin: array
            list of redshift bins
        Returns:
        --------
        Sij: array
            matter fluctuation amplitude in redshift bins
        r"""
        
        z_arr = np.linspace(0.05,2.5,500)
        nbins_T   = len(Z_bin)
        windows_T = np.zeros((nbins_T,len(z_arr)))
        for i, z_bin in enumerate(Z_bin):
            Dz = z_bin[1]-z_bin[0]
            z_arr_cut = z_arr[(z_arr > z_bin[0])*(z_arr < z_bin[1])]
            for k, z in enumerate(z_arr):
                if ((z>z_bin[0]) and (z<=z_bin[1])):
                    windows_T[i,k] = 1/Dz  
            
        default_cosmo_params = {'omega_b':cosmo['Omega_b']*cosmo['h']**2, 
                            'omega_cdm':cosmo['Omega_c']*cosmo['h']**2, 
                            'H0':cosmo['h']*100, 
                            'n_s':cosmo['n_s'], 
                            'sigma8': cosmo['sigma8'],
                            'output' : 'mPk'}
        
        return PySSC.Sij_alt_fullsky(z_arr, windows_T, order=2, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0)

    def sample_covariance_full_sky(self, Z_bin, Proxy_bin, NBinned_halo_bias, Sij):
        r"""
        returns the sample covariance matrix for cluster count
        Attributes:
        -----------
         Redshift_bin : list of lists
            list of redshift bins
        Proxy_bin : list of lists
            list of mass bins
        Binned_Abundance: array
            predicted abundance
        Binned_halo_bias: array
            predicted binned halo bias
            Sij: array
        matter fluctuation amplitude per redshift bin
        Returns:
        --------
        sample_covariance: array
            sample covariance for cluster abundance
            #uses the calculation of the fluctuation apmplitude Sij
        """
        index_LogM, index_Z =  np.meshgrid(np.arange(len(Proxy_bin)), np.arange(len(Z_bin)))
        index_Z_flatten = index_Z.flatten()
        len_mat = len(Z_bin) * len(Proxy_bin)
        cov_SSC = np.zeros([len_mat, len_mat])
        Nb = NBinned_halo_bias#np.multiply(Binned_Abundance, Binned_halo_bias)
        Nbij = np.tensordot(Nb, Nb)
        for i, Nbi in enumerate(Nb.flatten()):
           # if i%100==0: print(i)
            for j, Nbj in enumerate(Nb.flatten()):
                if i >= j:
                    index_z_i, index_z_j = index_Z_flatten.flatten()[i], index_Z_flatten.flatten()[j]
                    cov_SSC[i,j] = Nbi * Nbj * Sij[index_z_i, index_z_j]
                    cov_SSC[j,i] = cov_SSC[i,j]
        return cov_SSC
