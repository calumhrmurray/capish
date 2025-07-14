import numpy as np
import _completeness
import _mass_observable_relation
import _photometric_redshift
import _purity
import _selection
class ClusterCatalogue:
     
    def __init__( self , settings):
        """
        Initialize the ClusterCatalogue class with the given settings.
        """
        self.settings = settings
        return None
    
    def get_cluster_catalogue(self, log10m_true, z_true):

        
        if self.settings['cluster_catalogue']['add_completeness']=='True':
            a=1
            u = np.random.random(len(log10m_true))
            mask = u < _completeness.completeness(log10m_true, z_true)
            log10m_true, z_true = log10m_true[mask], z_true[mask]
            
        pivot_obs_z0 = float(self.settings['cluster_catalogue.mass_observable_relation']['pivot_obs_z0'])
        pivot_obs_log10m0 = float(self.settings['cluster_catalogue.mass_observable_relation']['pivot_obs_log10m0'])
        params_observable_mean = [float(k) for k in self.settings['cluster_catalogue.mass_observable_relation']['params_mean_obs'].split(', ')]
        params_observable_mean = [pivot_obs_log10m0, pivot_obs_z0] + params_observable_mean
        params_observable_stdd = [float(k) for k in self.settings['cluster_catalogue.mass_observable_relation']['params_stdd_obs'].split(', ')]
        params_observable_stdd = [pivot_obs_log10m0, pivot_obs_z0] + params_observable_stdd
        params_mWL_mean = [float(k) for k in self.settings['cluster_catalogue.mass_observable_relation']['params_mean_log10mWL'].split(', ')]
        params_mWL_stdd = [float(k) for k in self.settings['cluster_catalogue.mass_observable_relation']['params_stdd_log10mWL'].split(', ')]
        rho = float(self.settings['cluster_catalogue.mass_observable_relation']['rho_obs_mWL'])
        which = self.settings['cluster_catalogue.mass_observable_relation']['which_relation']

        MoR = _mass_observable_relation.Mass_observable_relation(params_observable_mean, params_observable_stdd, 
                                                                 params_mWL_mean, params_mWL_stdd, 
                                                                 rho, which = which)

        richness, log10mWL, z_true = MoR.generate_mWL_richness(log10m_true, z_true)
        
        if self.settings['cluster_catalogue']['add_photometric_redshift']=='True': 
            sigma_z0 = float(self.settings['cluster_catalogue.photometric_redshift']['sigma_z0'])
            z_obs = _photometric_redshift.photometric_redshift(z_true, sigma_z0)
            z_obs[z_obs < 0] = None
        else: 
            z_obs = z_true

        if self.settings['cluster_catalogue']['add_purity']=='True': 
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
        
        if self.settings['cluster_catalogue']['add_selection']=='True': 
            mask_selection = _selection.selection(richness, z_obs)
            richness, log10mWL, z_obs = richness[mask_selection], log10mWL[mask_selection], z_obs[mask_selection]

        return richness, log10mWL, z_obs


