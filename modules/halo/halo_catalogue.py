import numpy as np
from modules.halo import cosmology, halo_abundance
import pyccl as ccl
import modules.halo.halo_abundance_covariance as halo_abundance_covariance

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

def compute_Sij_matrix( cosmo, Z_bin_hybrid, f_sky = 1):
    hac = halo_abundance_covariance.Covariance_matrix()
    Sij_partialsky_exact_standard = hac.compute_theoretical_Sij(Z_bin_hybrid, cosmo, 
                                                                f_sky,
                                                                S_ij_type='full_sky_rescaled_approx', 
                                                                path=None)
    return Sij_partialsky_exact_standard

def compute_sigmaij_matrix( cosmo, z_grid, f_sky = 1):
    z_grid_center = np.array([(z_grid[i] + z_grid[i+1])/2 for i in range(len(z_grid)-1)])
    hac = halo_abundance_covariance.Covariance_matrix()
    sigmaij_partialsky_exact_standard = hac.compute_theoretical_sigmaij(z_grid_center, cosmo, f_sky)
    return sigmaij_partialsky_exact_standard

class HaloCatalogue:
     
    def __init__( self , settings ):
        """
        Initialize the HaloCatalogue class with the given settings.
        """

        # choose the hmf, cosmology, and bias model

        # this thing is currently named poorly!
        self.hmf = get_massfunc_from_config(settings['halo_catalogue'])
        self.bias = self.halobias_fct = get_bias_from_config(settings['halo_catalogue'])

        CosmologyObject = cosmology.Cosmology( hmf = self.hmf,
                                                bias_model = self.bias ) 
        # why clc, what does this mean?
        self.clc = halo_abundance.HaloAbundance( CosmologyObject = CosmologyObject , 
                                                    sky_area = float( settings['halo_catalogue']['sky_area'] ) )

        # set up the different grids
        self.logm_grid = np.linspace( float( settings['halo_catalogue']['log10m_min']),
                                      float( settings['halo_catalogue']['log10m_max']),
                                      int( settings['halo_catalogue']['n_mass_bins'] ) )
        self.dlogm_grid = self.logm_grid[1] - self.logm_grid[0]
        self.logm_grid_center = np.array([(self.logm_grid[i] + self.logm_grid[i+1])/2 for i in range(len(self.logm_grid)-1)])

        self.mass_definition = get_massdef_from_config(settings['halo_catalogue'])
        self.hmf = get_massfunc_from_config(settings['halo_catalogue'])

        # this could be done so much better
        self.hybrid = bool( settings['halo_catalogue']['hybrid'] )

        self.z_grid = np.linspace( float( settings['halo_catalogue']['z_min'] ),
                                    float( settings['halo_catalogue']['z_max'] ),
                                    int( settings['halo_catalogue']['n_redshift_bins'] ) )
        self.Z_hybrid = binning( self.z_grid )
        self.dz_grid = self.z_grid[1] - self.z_grid[0]
        self.z_grid_center = np.array([(self.z_grid[i] +self. z_grid[i+1])/2 for i in range(len(self.z_grid)-1)])

        self.SSC = bool( settings['halo_catalogue']['SSC'] )
        # setup the SSC stuff
        if bool( settings['halo_catalogue']['SSC'] ):
            ValueError("SSC is not implemented yet, please set SSC to False in the settings file.")
            # self.SSC = True
            # # fiducial cosmology for the SSC computation
            # cosmo = ccl.Cosmology( Omega_c = float( settings['halo_catalogue']['Omega_c_fiducial'] ), 
            #                        Omega_b = float( settings['halo_catalogue']['Omega_b_fiducial'] ), 
            #                        h = float( settings['halo_catalogue']['h_fiducial'] ), 
            #                        sigma8 = float( settings['halo_catalogue']['sigma_8_fiducial'] ), 
            #                        n_s=float( settings['halo_catalogue']['n_s_fiducial'] ) )
            # self.Sij_SSC = compute_Sij_matrix( cosmo, 
            #                                    self.Z_hybrid, 
            #                                    f_sky = float( settings['halo_catalogue']['f_sky'] ) )
            # self.sigmaij_SSC = compute_sigmaij_matrix( CosmologyObject.cosmo, 
            #                                             self.z_grid_center, 
            #                                             f_sky = float( settings['halo_catalogue']['f_sky'] ) )

        # hmf correction parameters
        self.Mstar = float( settings['halo_catalogue']['Mstar'] ) # in
        self.s = float( settings['halo_catalogue']['s'] ) # in log10
        self.q = float( settings['halo_catalogue']['q'] ) # in log10    
          
    def get_halo_catalogue( self, cosmo, return_Nth = False ):

            self.clc.set_cosmology( cosmo = cosmo )

            if (self.hybrid == False):

                #here, we compute the HMF grid
                self.clc.compute_multiplicity_grid_MZ(z_grid = z_grid_center, logm_grid = self.logm_grid_center)
                #we consider using the trapezoidal integral method here, given by int = dx(f(a) + f(b))/2
                # this is poorly named
                hmf_correction = self.hmf_correction(10 ** self.logm_grid_center, self.Mstar/cosmo['h'], self.s, self.q)
                dN_dzdlogMdOmega_center = self.clc.dN_dzdlogMdOmega * np.tile(hmf_correction, (len(z_grid_center), 1)).T

                if (self.SSC == True): 

                    self.clc.compute_halo_bias_grid_MZ(z_grid = z_grid_center, 
                                                logm_grid = self.logm_grid_center)
                    #generate deltas (log-normal probabilities)
                    cov_ln1_plus_delta_SSC = np.log(1 + self.sigmaij_SSC)
                    mean = - 0.5 * cov_ln1_plus_delta_SSC.diagonal()
                    ln1_plus_delta_SSC = np.random.multivariate_normal(mean=mean , cov=cov_ln1_plus_delta_SSC)
                    delta = (np.exp(ln1_plus_delta_SSC) - 1)
                    delta_h = self.clc.halo_biais * delta
                    delta_h = np.where(delta_h < -1, -1, delta_h)
                    corr = 1 + delta_h
                else: corr = 1

                Omega_z = np.tile(self.dOmega(z_grid_center), (len(z_grid_center), 1)).T
                    
                Nobs = np.random.poisson(Omega_z * dN_dzdlogMdOmega_center * self.dlogm_grid * dz_grid * corr)
                Nobs_flatten = Nobs.flatten()
                Z_grid_center, Logm_grid_center = np.meshgrid(z_grid_center, self.logm_grid_center)
                Z_grid_center_flatten, Logm_grid_center_flatten = Z_grid_center.flatten(), Logm_grid_center.flatten()

                log10mass = [logm_grid_i for logm_grid_i, count in zip(Logm_grid_center_flatten, Nobs_flatten) for _ in range(count)]
                redshift = [z_grid_i for z_grid_i, count in zip(Z_grid_center_flatten, Nobs_flatten) for _ in range(count)]

                grid = {"N_th": Omega_z * dN_dzdlogMdOmega_center * dlogm_grid * dz_grid, 
                        "z_grid_center":self.z_grid_center, 
                        "logm_grid_center":self.logm_grid_center}
            
            elif self.hybrid:
                
                Z_edges_hybrid = self.Z_hybrid
                Z_bin_hybrid = [[Z_edges_hybrid[i], Z_edges_hybrid[i+1]] for i in range(len(Z_edges_hybrid)-1)]
                
                #ensure that the redshift grid matches SSC redshift grid values
                z_grid = []
                z_grid.append(Z_edges_hybrid[0])
                for i in range(len(Z_edges_hybrid)-1):
                    x = list(np.linspace(Z_edges_hybrid[i], Z_edges_hybrid[i+1], 50))
                    z_grid.extend(x[1:-1])
                    z_grid.append(Z_edges_hybrid[i+1])
                z_grid = np.array(z_grid)
                dz_grid = z_grid[1] - z_grid[0]
                
                #here, we compute the HMF grid
                self.clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = self.logm_grid)
                #we consider using the trapezoidal integral method here, given by int = dx(f(a) + f(b))/2
                dN_dzdlogMdOmega_center = (self.clc.dN_dzdlogMdOmega[:-1] + self.clc.dN_dzdlogMdOmega[1:]) / 2
                # this is calculated earlier? why again?
                logm_grid_center = np.array([(logm_grid[i] + logm_grid[i+1])/2 for i in range(len(self.logm_grid)-1)])
                hmf_correction = self.hmf_correction(10**logm_grid_center, self.Mstar/cosmo['h'], self.s, self.q)
                dN_dzdlogMdOmega_center *= np.tile(hmf_correction, (len(z_grid), 1)).T

                if self.SSC: 
                    self.clc.compute_halo_bias_grid_MZ(z_grid = z_grid, logm_grid = self.logm_grid)
                    halo_bias_center = (self.clc.halo_biais[:-1] + self.clc.halo_biais[1:]) / 2
                    #generate deltas in redshift bins (log-normal probabilities)
                    cov_ln1_plus_delta_SSC = np.log(1 + self.Sij_SSC)
                    mean = - 0.5 * cov_ln1_plus_delta_SSC.diagonal()
                    ln1_plus_delta_SSC = np.random.multivariate_normal(mean=mean , cov=cov_ln1_plus_delta_SSC)
                    delta = (np.exp(ln1_plus_delta_SSC) - 1)

                N_obs = np.zeros([len(Z_bin_hybrid), len(logm_grid_center)])
                N_th = np.zeros([len(Z_bin_hybrid), len(logm_grid_center)])
                log10mass, redshift = [], []
                
                for i, redshift_range in enumerate(Z_bin_hybrid):
                    
                    mask = (z_grid >= redshift_range[0])*(z_grid <= redshift_range[1])
                    dNdm  = self.dOmega(z_grid[mask]) * np.trapz(dN_dzdlogMdOmega_center[:,mask], z_grid[mask], axis=1)
                    pdf   = self.dOmega(z_grid[mask]) * dN_dzdlogMdOmega_center[:,mask]
                    cumulative = np.cumsum(dz_grid * pdf, axis = 1)  
                    
                    if self.add_SSC == True:
                        integrand = self.dOmega(z_grid[mask]) * halo_bias_center * dN_dzdlogMdOmega_center
                        bdNdm = np.trapz(integrand[:,mask], z_grid[mask])
                        bias = np.array(bdNdm)/np.array(dNdm)
                        delta_h = bias * delta[i]
                        delta_h = np.where(delta_h < -1, -1, delta_h) #we ensure that deltah = b*delta is > 1
                        corr = (1 + delta_h)
                    else: 
                        corr = 1

                    N_obs[i,:] = np.random.poisson(dNdm * dlogm_grid * corr)
                    N_th[i,:] = dNdm * dlogm_grid #we generate the observed count
                    N_sample_obs_zbins = N_obs[i,:]
                    
                    for j in range(len(logm_grid_center)):
                        log10mass.extend(list(np.zeros(int(N_sample_obs_zbins[j]))+logm_grid_center[j])) #masses
                        cumulative_rand = (cumulative[j][-1]-cumulative[j][0])*np.random.random(int(N_sample_obs_zbins[j]))+cumulative[j][0]
                        redshift.extend(list(np.interp(cumulative_rand, cumulative[j], z_grid[mask]))) #redshifts
                        
                grid = {"N_th":N_th, "z_grid":z_grid, "logm_grid":logm_grid, "logm_grid_center": logm_grid_center}

            #log10mass_return = np.log(10**np.array(log10mass)/(10**14))
            log10mass_return = log10mass
            if return_Nth:
                return grid, log10mass_return, np.array(redshift)
            else:
                return log10mass_return, np.array(redshift)

    def hmf_correction( self , M , Mstar , s   , q ):
        return s * np.log10( M / Mstar ) + q