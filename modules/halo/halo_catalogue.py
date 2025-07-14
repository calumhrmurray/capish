import numpy as np
from modules.halo import halo_abundance
import pyccl as ccl
import pickle
from pathlib import Path

current_file_path = str(Path(__file__).resolve()).split('halo_catalogue.py')[0]+'/SSC/'

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()
    
def load_pickle(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

def binning(corner): 
    return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]


# move these maps elsewhere 
MASSFUNC_MAP = {
    "Despali16": ccl.halos.hmfunc.MassFuncDespali16,
    "Tinker10": ccl.halos.hmfunc.MassFuncTinker10,
    # add all of the possible ccl hmf...
}

BIAS_MAP = {
    "Tinker10": ccl.halos.hbias.tinker10.HaloBiasTinker10,
    # Add more as needed
}

def get_massdef_from_config(settings):
    mass_def_overdensity_type = settings["mass_def_overdensity_type"]
    mass_def_overdensity_delta = settings["mass_def_overdensity_delta"]
    return ccl.halos.massdef.MassDef( mass_def_overdensity_delta , 
                                      mass_def_overdensity_type )

def get_massfunc_from_config(settings):
    mass_func_cls = MASSFUNC_MAP[settings["hmf_name"]]
    massdef = get_massdef_from_config( settings )
    return mass_func_cls( mass_def = massdef )

def get_bias_from_config(settings):
    bias_model_name = settings.get("bias_model", "Tinker10")
    bias_cls = BIAS_MAP[bias_model_name]
    massdef = get_massdef_from_config(settings)
    return bias_cls(mass_def=massdef, mass_def_strict=True)

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

class HaloCatalogue:
     
    def __init__( self , settings ):
        """
        Initialize the HaloCatalogue class with the given settings.
        """

        # choose the hmf, cosmology, and bias model
        self.hmf = get_massfunc_from_config(settings['halo_catalogue'])
        self.bias = get_bias_from_config(settings['halo_catalogue'])
        self.sky_area = float( settings['halo_catalogue']['sky_area'] )
        def dOmega_fct(z): return self.sky_area #steradians
        self.dOmega = dOmega_fct
        self.fsky = self.sky_area/(4*np.pi) #%

        # set up the different grids
        ## mass grid
        self.logm_grid = np.linspace( float( settings['halo_catalogue']['log10m_min']),
                                      float( settings['halo_catalogue']['log10m_max']),
                                      int( settings['halo_catalogue']['n_mass_bins'] ) )
        self.dlogm_grid = self.logm_grid[1] - self.logm_grid[0]
        self.logm_grid_center = (self.logm_grid[:-1] + self.logm_grid[1:]) / 2
        #redshift grid
        self.z_grid = np.linspace( float( settings['halo_catalogue']['z_min'] ),
                                    float( settings['halo_catalogue']['z_max'] ),
                                    int( settings['halo_catalogue']['n_redshift_bins'] ) )
        self.dz_grid = self.z_grid[1] - self.z_grid[0]
        self.z_grid_center = (self.z_grid[:-1] + self.z_grid[1:]) / 2
        #very important for later !
        Z, L = np.meshgrid(self.z_grid_center, self.logm_grid_center)
        self.Z_grid_center_flatten = Z.ravel()
        self.Logm_grid_center_flatten = L.ravel()

        self.mass_definition = get_massdef_from_config(settings['halo_catalogue'])
        self.hmf = get_massfunc_from_config(settings['halo_catalogue'])

        self.SSC = str2bool(settings['halo_catalogue']['SSC'])
        self.recompute_SSC_fiducial = str2bool(settings['halo_catalogue']['recompute_SSC_ficucial'])
        self.save_new_SSC_fiducial = str2bool(settings['halo_catalogue']['save_new_SSC_fiducial'])
        
        # setup the SSC stuff
        if self.SSC:
            zmin = float( settings['halo_catalogue']['z_min'])
            zmax = float( settings['halo_catalogue']['z_max'])  
            nzbins =  int( settings['halo_catalogue']['n_redshift_bins'] )
            filename = str(settings['halo_catalogue']['name_sigma2ij_fullsky_file']).format(zmin, zmax, nzbins)
            filename = current_file_path + filename
            if self.recompute_SSC_fiducial:
                # fiducial cosmology for the SSC computation, will not be run at every step of CAPISH
                cosmo_ccl_fid = ccl.Cosmology( Omega_c = float( settings['halo_catalogue']['Omega_c_fiducial'] ), 
                                               Omega_b = float( settings['halo_catalogue']['Omega_b_fiducial'] ), 
                                               h = float( settings['halo_catalogue']['h_fiducial'] ), 
                                               sigma8 = float( settings['halo_catalogue']['sigma_8_fiducial'] ), 
                                               n_s=float( settings['halo_catalogue']['n_s_fiducial'] ) )

                try: #this step will recquire PySSC, will fail otherwise
                    HaloAbundanceObject = halo_abundance.HaloAbundance()
                    sigma2ij_SSC_fullsky = HaloAbundanceObject.compute_theoretical_sigma2ij_fullsky(cosmo_ccl_fid, 
                                                                                self.z_grid_center)
                    self.sigmaij_SSC = sigma2ij_SSC_fullsky/self.fsky
                except:
                    raise ValueError("should install PySSC!")
                
                if self.save_new_SSC_fiducial:
                    sigma2ij_SSC_tosave = {'z_grid_center': self.z_grid_center,
                                      'sigma2ij_SSC_fullsky': sigma2ij_SSC_fullsky}
                    save_pickle(sigma2ij_SSC_tosave, filename)
                
            else: 
                sigmaij_SSC_file = np.load(filename, allow_pickle=True)
                self.sigmaij_SSC = sigmaij_SSC_file['sigma2ij_SSC_fullsky']/self.fsky 
                check_z_grid = 0
                for x, y in zip(sigmaij_SSC_file['z_grid_center'], self.z_grid_center):
                    if x != y: check_z_grid += 1
                if check_z_grid!=0: raise ValueError("Mismatch - should install PySSC!")
            
        # hmf correction parameters
        self.Mstar = float( settings['halo_catalogue']['Mstar'] ) # in
        self.s = float( settings['halo_catalogue']['s'] ) # in log10
        self.q = float( settings['halo_catalogue']['q'] ) # in log10 
          
    def get_halo_catalogue(self, cosmo, return_Nth = False ):

            #recall that here cosmo is now a CCL object !
            HaloAbundanceObject = halo_abundance.HaloAbundance( CCLCosmologyObject = cosmo, 
                                                                     CCLHmf = self.hmf,
                                                                     CCLBias = self.bias,
                                                                     sky_area = self.sky_area )

            #here, we compute the HMF grid
            HaloAbundanceObject.compute_multiplicity_grid_MZ(z_grid = self.z_grid_center, logm_grid = self.logm_grid_center)
            #we consider using the trapezoidal integral method here, given by int = dx(f(a) + f(b))/2

            hmf_correction = self.hmf_correction(10 ** self.logm_grid_center, self.Mstar/cosmo['h'], self.s, self.q)
            dN_dzdlogMdOmega_center = HaloAbundanceObject.dN_dzdlogMdOmega * np.tile(hmf_correction, (len(self.z_grid_center), 1)).T

            if self.SSC: 
                #at this stage, need to store self.sigmaij_SSC computed from PySSC
                HaloAbundanceObject.compute_halo_bias_grid_MZ(z_grid = self.z_grid_center, 
                                                                   logm_grid = self.logm_grid_center)
                #generate deltas (log-normal probabilities)
                cov_ln1_plus_delta_SSC = np.log(1 + self.sigmaij_SSC)
                mean = - 0.5 * cov_ln1_plus_delta_SSC.diagonal()
                ln1_plus_delta_SSC = np.random.multivariate_normal(mean=mean , cov=cov_ln1_plus_delta_SSC)
                delta = (np.exp(ln1_plus_delta_SSC) - 1)
                delta_h = HaloAbundanceObject.halo_biais * delta
                delta_h = np.where(delta_h < -1, -1, delta_h)
                corr = 1 + delta_h
            else: corr = 1

            Omega_z = np.tile(self.dOmega(self.z_grid_center), (len(self.z_grid_center), 1)).T
            Nth = Omega_z * dN_dzdlogMdOmega_center * self.dlogm_grid * self.dz_grid * corr
            Nobs = np.random.poisson(Nth)
            Nobs_flatten = Nobs.ravel()
            log10mass = np.repeat(self.Logm_grid_center_flatten, Nobs_flatten)
            redshift = np.repeat(self.Z_grid_center_flatten, Nobs_flatten)

            grid = {"N_th": Omega_z * dN_dzdlogMdOmega_center * self.dlogm_grid * self.dz_grid, 
                    "z_grid_center":self.z_grid_center, 
                    "logm_grid_center":self.logm_grid_center}
        
            if return_Nth:
                return grid, log10mass, np.array(redshift)
            else:
                return log10mass, np.array(redshift)

    def hmf_correction( self , M , Mstar , s   , q ):
        return s * np.log10( M / Mstar ) + q