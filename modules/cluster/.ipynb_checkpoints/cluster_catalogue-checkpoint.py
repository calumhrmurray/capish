import numpy as np
from . import _completeness
from . import _halo_observable_relation
from . import _purity
from . import _selection
    
class ClusterCatalogue:
     
    def __init__( self , default_config):
        """
        Initialize the ClusterCatalogue class with the given settings.
        """
        self.default_config = default_config
        return None
    
    def get_cluster_catalogue(self, log10m_true, z_true, config_new):

        params_completeness = [float(k) for k in config_new['cluster_catalogue']['params_completeness'].split(', ')]
        params_purity = [float(k) for k in config_new['cluster_catalogue']['params_purity'].split(', ')]
        
        if config_new['cluster_catalogue']['add_completeness']=='True':
            u = np.random.random(len(log10m_true))
            mask = u < _completeness.completeness(log10m_true, z_true, params = params_completeness)
            log10m_true, z_true = log10m_true[mask], z_true[mask]

        MoR = _halo_observable_relation.HaloToObservables(config_new)
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


