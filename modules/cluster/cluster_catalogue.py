import numpy as np
import _completeness
import _halo_observable_relation
import _purity
import _selection
    
class ClusterCatalogue:
     
    def __init__( self , default_config):
        """
        Initialize the ClusterCatalogue class with the given settings.
        """
        self.default_config = default_config
        return None
    
    def get_cluster_catalogue(self, log10m_true, z_true, config_new):
        
        if config_new['cluster_catalogue']['add_completeness']=='True':
            a=1
            u = np.random.random(len(log10m_true))
            mask = u < _completeness.completeness(log10m_true, z_true)
            log10m_true, z_true = log10m_true[mask], z_true[mask]

        parameters = config_new['parameters']
        pivot_obs_z0 = float(parameters['pivot_obs_z0'])
        pivot_obs_log10m0 = float(parameters['pivot_obs_log10m0'])
        params_observable_mean = [float(parameters['params_mean_obs_mu0']), float(parameters['params_mean_obs_muz']), float(parameters['params_mean_obs_mulog10m'])]
        params_observable_stdd = [float(parameters['params_stdd_obs_mu0']), float(parameters['params_stdd_obs_muz']), float(parameters['params_stdd_obs_mulog10m'])]
        params_observable_mean = [pivot_obs_log10m0, pivot_obs_z0] + params_observable_mean
        params_observable_stdd = [pivot_obs_log10m0, pivot_obs_z0] + params_observable_stdd
        params_mWL_mean = [float(parameters['params_mean_log10mWL_aWL']), float(parameters['params_mean_log10mWL_bWL'])]
        params_mWL_stdd = [float(parameters['params_stdd_log10mWLgal']), float(parameters['params_stdd_log10mWLint'])]
        rho = float(parameters['rho_obs_mWL'])
        which = config_new['cluster_catalogue.mass_observable_relation']['which_relation']

        add_photoz = True if config_new['cluster_catalogue']['add_photometric_redshift']=='True' else False
        sigma_z0 = float(config_new['cluster_catalogue.photometric_redshift']['sigma_z0'])
        MoR = _halo_observable_relation.HaloToObservables(params_observable_mean, params_observable_stdd, 
                                                                 params_mWL_mean, params_mWL_stdd, 
                                                                 rho, which_mass_richness_rel = which, 
                                                                 add_photoz=add_photoz, photoz_params=sigma_z0)
        
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
                    purity_grid[i, j] = _purity.purity(richness_center[i], z_obs_center[j])
    
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


