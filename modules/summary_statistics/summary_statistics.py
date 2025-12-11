import numpy as np
import pyccl as ccl
import io
from scipy import stats
import configparser
import modules.utils as utils

def clone_config(cfg):
    s = io.StringIO()
    cfg.write(s)
    s.seek(0)
    new_cfg = configparser.ConfigParser()
    new_cfg.read_file(s)
    return new_cfg

class SummaryStatistics:
     
    def __init__( self , default_config):
        
        self.default_config = default_config

        self.richness_edges = np.array(list(map(float, default_config['summary_statistics']['richness_edges'].split(','))), dtype=float)
        self.redshift_edges = np.array(list(map(float, default_config['summary_statistics']['redshift_edges'].split(','))), dtype=float)

        Omega_c_fid = float(default_config['halo_catalogue']['Omega_c_fiducial'])
        Omega_b_fid = float(default_config['halo_catalogue']['Omega_b_fiducial'])
        sigma8_fid = float(default_config['halo_catalogue']['sigma_8_fiducial'])
        h_fid = float(default_config['halo_catalogue']['h_fiducial'])
        ns_fid = float(default_config['halo_catalogue']['n_s_fiducial'])

        self.Gamma = float(default_config['summary_statistics']['Gamma'])
        
        self.use_stacked_sigma_Mwl_gal = default_config['summary_statistics']['use_stacked_sigma_Mwl_gal']
        self.use_stacked_sigma_Mwl_int = default_config['summary_statistics']['use_stacked_sigma_Mwl_int']
        self.theory_sigma_Mwl_gal = default_config['cluster_catalogue']['theory_sigma_Mwl_gal']
        self.sigma_Mwl_gal = float(default_config['parameters']['sigma_Mwl_gal'])
        self.sigma_Mwl_int = float(default_config['parameters']['sigma_Mwl_int'])

        if self.use_stacked_sigma_Mwl_gal == 'True':
            if self.theory_sigma_Mwl_gal == 'True':
                from modules.cluster.cluster_catalogue import ClusterCatalogue
                default_config_ClusterCatalogue = clone_config(default_config)
                default_config_ClusterCatalogue['cluster_catalogue']['theory_sigma_Mwl_gal']='True'
                default_config_ClusterCatalogue['cluster_catalogue']['recompute_theory_sigma_Mwl_gal']='False'
                self.cluster_catalogue_class = ClusterCatalogue(default_config_ClusterCatalogue)
                self.sigma_log10Mwl_gal_interp = self.cluster_catalogue_class.sigma_log10Mwl_gal_interp

            else: 
                def sigma_log10Mwl_gal_interp(log10m, z): return self.sigma_Mwl_gal
                self.sigma_log10Mwl_gal_interp = sigma_log10Mwl_gal_interp
        else: 
            def sigma_log10Mwl_gal_interp(log10m, z): return 0
            self.sigma_log10Mwl_gal_interp = sigma_log10Mwl_gal_interp

        if self.use_stacked_sigma_Mwl_int == 'True':
            def sigma_log10Mwl_int_interp(log10m, z): return self.sigma_Mwl_int
            self.sigma_log10Mwl_int_interp = sigma_log10Mwl_int_interp
        else: 
            def sigma_log10Mwl_int_interp(log10m, z): return 0
            self.sigma_log10Mwl_int_interp = sigma_log10Mwl_int_interp

        cosmo_fid = ccl.Cosmology( Omega_c = Omega_c_fid, Omega_b = Omega_b_fid, 
                                  h = h_fid, sigma8 = sigma8_fid, n_s= ns_fid)
        
        z_l_array = np.linspace(0.03, 3, 300)
        W_zl = utils.lensing_weights(cosmo_fid, z_l_array, z_s_max=3.0, n_zs=500, sigma_e_const=0.3)
        def W_zl_f(z_l): return np.interp(z_l, z_l_array, W_zl)
        self.W_zl_f = W_zl_f
        
        return None

    def get_summary_statistics(self, richness, log10mWL, z_obs, config_new):

        if config_new['summary_statistics']['summary_statistic'] == 'binned_count_mean_mass':

            nr = len(self.richness_edges) - 1
            nz = len(self.redshift_edges) - 1
            if len(richness) == 0:
                count_stat = np.zeros((nr, nz))
                mean_mass_stat = np.zeros((nr, nz))
                mean_mass_stat_scatter = np.zeros((nr, nz))
                #because stats.binned_statistic_2d() only works with non-empty lists
            else:
                Wz = self.W_zl_f(z_obs)
                bins = [self.richness_edges, self.redshift_edges]
                count_stat, x_edges, y_edges, _ = stats.binned_statistic_2d(richness, z_obs, None, statistic='count',bins=bins)
                mask = log10mWL != None
                mass_gamma = np.array((10 ** log10mWL[mask]) ** self.Gamma,dtype=float)
                sum_w_mass_gamma_stat, _, _, _ = stats.binned_statistic_2d(richness[mask], z_obs[mask], Wz[mask] * mass_gamma, statistic='sum', bins=bins)
                sum_w_stat, _, _, _ = stats.binned_statistic_2d(richness, z_obs, Wz, statistic='sum', bins=bins)
                mean_mass_stat = np.log10((sum_w_mass_gamma_stat/sum_w_stat)**(1/self.Gamma))
                # Set NaN masses (from empty bins) to zero
                mean_mass_stat = np.nan_to_num(mean_mass_stat, nan=0.0)
                z_centers = [(self.redshift_edges[i+1] + self.redshift_edges[i])/2 for i in range(len(self.redshift_edges)-1)]
                richness_centers = [(self.richness_edges[i+1] + self.richness_edges[i])/2 for i in range(len(self.richness_edges)-1)]
                for k in range(nz):
                    z_centers_duplicate = np.linspace(z_centers[k], z_centers[k], nr)
                    WLdispersion_gal = self.sigma_log10Mwl_gal_interp(mean_mass_stat[:,k], z_centers_duplicate)
                    WLdispersion_int = self.sigma_log10Mwl_int_interp(mean_mass_stat[:,k], z_centers_duplicate)
                    WLdispersion = np.sqrt(WLdispersion_gal**2 + WLdispersion_int**2)
                    print(WLdispersion)
                    mean_mass_stat[:,k] = mean_mass_stat[:,k] + np.random.randn(len(richness_centers)) * WLdispersion * 1/np.sqrt(count_stat[:,k])

            return count_stat, mean_mass_stat 
            
        if config_new['summary_statistics']['summary_statistic'] == '3d_count':
            mask0 = log10mWL != None
            threed_hist = np.zeros([len(self.richness_edges)-1, len(self.redshift_edges)-1, len(self.log10mWL_edges)-1])
            twod_bins = [self.richness_edges, self.redshift_edges,]
            for i in range(len(self.log10mWL_edges)-1):
                mask_mass = (log10mWL[mask0] > self.log10mWL_edges[i]) * (log10mWL[mask0] < self.log10mWL_edges[i+1])
                threed_hist[:,:,i] = np.histogram2d(richness[mask0][mask_mass], z_obs[mask0][mask_mass],bins=twod_bins)[0]
            return threed_hist

















            