import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pyccl as ccl
from . import _completeness
from . import _halo_observable_relation
from . import _purity
from . import _selection
from .. import utils

class ClusterCatalogue:
     
    def __init__( self , default_config):
        """
        Initialize the ClusterCatalogue class with the given settings.
        """
        self.default_config = default_config

        # set up the different grids
        ## mass grid

        log10m_grid = np.linspace( float( default_config['halo_catalogue']['log10m_min']),
                                      float( default_config['halo_catalogue']['log10m_max']),
                                      int( 10 ) )
        log10m_grid_center = (log10m_grid[:-1] + log10m_grid[1:]) / 2
        #redshift grid
        z_grid = np.linspace( float( default_config['halo_catalogue']['z_min'] ),
                                    float( default_config['halo_catalogue']['z_max'] ),
                                    int( 10 ) )
        z_grid_center = (z_grid[:-1] + z_grid[1:]) / 2
        
        cosmo_ccl_fid = ccl.Cosmology( Omega_c = float( default_config['halo_catalogue']['Omega_c_fiducial'] ), 
                                           Omega_b = float( default_config['halo_catalogue']['Omega_b_fiducial'] ), 
                                           h = float( default_config['halo_catalogue']['h_fiducial'] ), 
                                           sigma8 = float( default_config['halo_catalogue']['sigma_8_fiducial'] ), 
                                           n_s=float( default_config['halo_catalogue']['n_s_fiducial'] ) )

        mass_def = default_config['halo_catalogue']["mass_def_overdensity_type"]
        delta = int(default_config['halo_catalogue']["mass_def_overdensity_delta"])

        if self.default_config['cluster_catalogue']['theory_sigma_Mwl'] == 'True':

            sigma_log10M = utils.model_error_log10m_one_cluster(log10m_grid, z_grid, 
                                                                cosmo_ccl_fid, 
                                                               Rmin=1, Rmax=3, 
                                                               ngal_arcmin2=25, shape_noise=0.25,
                                                               delta=delta, mass_def=mass_def,
                                                               cM = 'Duffy08')
            
            self.log10m_grid, self.z_grid, self.sigma_log10M = log10m_grid, z_grid, sigma_log10M

            sigma_interp = RegularGridInterpolator((self.log10m_grid, self.z_grid),
                            self.sigma_log10M, bounds_error=False, fill_value=np.nan)

            def sigma_log10M_interp(log10m_array, z_array):
                points = np.column_stack([log10m_array, z_array])
                return sigma_interp (points)
            self.sigma_log10M_interp = sigma_log10M_interp

        else: 
            
            self.log10m_grid, self.z_grid, self.sigma_log10M = log10m_grid, z_grid, None
            self.sigma_log10M_interp = None
            
        return None
    
    def get_cluster_catalogue(self, log10m_true, z_true, config_new):

        params_completeness = [float(k) for k in config_new['cluster_catalogue']['params_completeness'].split(', ')]
        params_purity = [float(k) for k in config_new['cluster_catalogue']['params_purity'].split(', ')]
        
        if config_new['cluster_catalogue']['add_completeness']=='True':
            u = np.random.random(len(log10m_true))
            mask = u < _completeness.completeness(log10m_true, z_true, params = params_completeness)
            log10m_true, z_true = log10m_true[mask], z_true[mask]

        MoR = _halo_observable_relation.HaloToObservables(config_new, sigma_log10mWL_model = self.sigma_log10M_interp)
        richness, log10mWL, z_obs = MoR.generate_observables_from_halo(log10m_true, z_true)
        z_obs[z_obs < 0] = None

        if config_new['cluster_catalogue']['add_purity']=='True': 
            richness_edges = np.logspace(np.log10(20), np.log10(300), 70) 
            richness_center = 0.5 * (richness_edges[:-1] + richness_edges[1:])
            z_obs_edges = np.linspace(0, 2, 70)
            z_obs_center = 0.5 * (z_obs_edges[:-1] + z_obs_edges[1:])
            hist, xedges, yedges = np.histogram2d(richness, z_obs, bins=[richness_edges, z_obs_edges])
            Ntrue = hist 
    
            purity_grid = np.zeros_like(hist)
            for i in range(len(richness_center)):
                for j in range(len(z_obs_center)):
                    purity_grid[i, j] = _purity.purity(richness_center[i], z_obs_center[j], params = params_purity)
    
            Nfake_sampled = np.random.poisson(Ntrue * (1 - purity_grid))
            fake_richness, fake_z_obs = [], []
            
            for i in range(len(richness_edges) - 1):
                for j in range(len(z_obs_edges) - 1):
                    n_fake = Nfake_sampled[i, j]
                    if n_fake > 0:
                        richness_bin = np.random.uniform(richness_edges[i], richness_edges[i+1], size=n_fake)
                        z_bin = np.random.uniform(z_obs_edges[j], z_obs_edges[j+1], size=n_fake)
                        fake_richness.extend(richness_bin)
                        fake_z_obs.extend(z_bin)

            fake_log10mWL = [None] * len(fake_richness)
            richness = np.array(list(richness) + fake_richness)
            log10mWL = np.array(list(log10mWL) + fake_log10mWL)
            z_obs = np.array(list(z_obs) + fake_z_obs)
        
        if config_new['cluster_catalogue']['add_selection']=='True': 
            mask_selection = _selection.selection(richness, z_obs)
            richness, log10mWL, z_obs = richness[mask_selection], log10mWL[mask_selection], z_obs[mask_selection]

        return richness, log10mWL, z_obs
