import numpy as np
import pyccl as ccl
from scipy import stats

def nz_chang2013(z, alpha=2.0, beta=1.5, z0=0.5):
    return z**alpha * np.exp(-(z / z0)**beta)

def sigma_epsilon(z, sigma_e=0.3):
    return sigma_e * np.ones(len(z))  # constant or e.g., sigma_e * (1 + z)

def sigma_crit(cosmo, z_l, z_s):
    return ccl.sigma_critical(cosmo, a_lens=1/(1+z_l), a_source=1/(1+z_s))

def lensing_weights(cosmo, z_l_array, z_s_max=5.0, n_zs=500, sigma_e_const=0.3):
    """Compute W(z_l) for an array of lens redshifts using np.trapz."""
    z_l_array = np.atleast_1d(z_l_array)
    weights = np.zeros_like(z_l_array)

    # Source redshift grid
    z_s = np.linspace(0.01, z_s_max, n_zs)
    n_z = nz_chang2013(z_s)
    sigma_e = sigma_epsilon(z_s, sigma_e_const)

    for i, z_l in enumerate(z_l_array):
        mask = z_s > z_l
        z_s_masked = z_s[mask]
        n_z_masked = n_z[mask]
        sigma_e_masked = sigma_e[mask]

        if len(z_s_masked) == 0:
            weights[i] = 0.0
            continue

        sigma_crit_vals = sigma_crit(cosmo, z_l, z_s_masked)
        sigma_crit_sq_inv = 1.0 / sigma_crit_vals**2
        integrand = n_z_masked * sigma_crit_sq_inv / sigma_e_masked**2

        numerator = np.trapz(integrand, z_s_masked)
        denominator = np.trapz(n_z_masked, z_s_masked)
        weights[i] = numerator / denominator if denominator > 0 else 0.0
    return weights

class SummaryStatistics:
     
    def __init__( self , default_config):
        
        self.default_config = default_config

        if str(default_config['summary_statistics']['richness_edges']) == 'None':
            log10lambda_low = np.log10(float(default_config['summary_statistics']['richness_lower_limit']))
            log10lambda_up = np.log10(float(default_config['summary_statistics']['richness_upper_limit']))
            n_bins_lambda = int(default_config['summary_statistics']['n_bins_richness'])
            self.richness_edges = np.logspace(log10lambda_low, log10lambda_up, n_bins_lambda+1)
            self.richness_centers = 0.5 * (self.richness_edges[:-1] + self.richness_edges[1:])
        else: 
            self.richness_edges = np.array([float(k) for k in default_config['summary_statistics']['richness_edges'].split(', ')])
            self.richness_centers = 0.5 * (self.richness_edges[:-1] + self.richness_edges[1:])
        
        if str(default_config['summary_statistics']['redshift_edges']) == 'None':
            redshift_low = float(default_config['summary_statistics']['redshift_lower_limit'])
            redshift_up = float(default_config['summary_statistics']['redshift_upper_limit'])
            n_bins_redshift = int(default_config['summary_statistics']['n_bins_redshift'])
            self.redshift_edges = np.linspace(redshift_low, redshift_up, n_bins_redshift+1)
            self.redshift_centers = 0.5 * (self.redshift_edges[:-1] + self.redshift_edges[1:])
        else: 
            self.redshift_edges = np.array([float(k) for k in default_config['summary_statistics']['redshift_edges'].split(', ')])
            self.redshift_centers = 0.5 * (self.redshift_edges[:-1] + self.redshift_edges[1:])

        if str(default_config['summary_statistics']['log10mWL_edges']) == 'None':
            log10mWL_low = float(default_config['summary_statistics']['log10mWL_lower_limit'])
            log10mWL_up = float(default_config['summary_statistics']['log10mWL_upper_limit'])
            n_bins_log10mWL = int(default_config['summary_statistics']['n_bins_log10mWL'])
            self.log10mWL_edges = np.linspace(log10mWL_low, log10mWL_up, n_bins_log10mWL+1)
            self.log10mWL_centers = 0.5 * (self.log10mWL_edges[:-1] + self.log10mWL_edges[1:])
        else: 
            self.log10mWL_edges = np.array([float(k) for k in default_config['summary_statistics']['log10mWL_edges'].split(', ')])
            self.log10mWL_centers = 0.5 * (self.log10mWL_edges[:-1] + self.log10mWL_edges[1:])

        Omega_c_fid = float(default_config['halo_catalogue']['Omega_c_fiducial'])
        Omega_b_fid = float(default_config['halo_catalogue']['Omega_b_fiducial'])
        sigma8_fid = float(default_config['halo_catalogue']['sigma_8_fiducial'])
        h_fid = float(default_config['halo_catalogue']['h_fiducial'])
        ns_fid = float(default_config['halo_catalogue']['n_s_fiducial'])

        self.Gamma = float(default_config['summary_statistics']['Gamma'])
        
        cosmo_fid = ccl.Cosmology( Omega_c = Omega_c_fid, Omega_b = Omega_b_fid, 
                                  h = h_fid, sigma8 = sigma8_fid, n_s= ns_fid)
        
        z_l_array = np.linspace(0.03, 3, 300)
        W_zl = lensing_weights(cosmo_fid, z_l_array, z_s_max=3.0, n_zs=500, sigma_e_const=0.3)
        def W_zl_f(z_l): return np.interp(z_l, z_l_array, W_zl)
        self.W_zl_f = W_zl_f
        
        return None

    def get_summary_statistics(self, richness, log10mWL, z_obs, config_new):

        if config_new['summary_statistics']['summary_statistic'] == 'binned_count_mean_mass':

            Wz = self.W_zl_f(z_obs)
            bins = [self.richness_edges, self.redshift_edges,]
            count_stat, x_edges, y_edges, _ = stats.binned_statistic_2d(richness, z_obs, None, statistic='count',bins=bins)
            mask = log10mWL != None
            mass_gamma = np.array((10 ** log10mWL[mask]) ** self.Gamma,dtype=float)
            sum_w_mass_gamma_stat, _, _, _ = stats.binned_statistic_2d(richness[mask], z_obs[mask], Wz[mask] * mass_gamma, statistic='sum', bins=bins)
            sum_w_stat, _, _, _ = stats.binned_statistic_2d(richness, z_obs, Wz, statistic='sum', bins=bins)
            return count_stat, np.log10((sum_w_mass_gamma_stat/sum_w_stat)**(1/self.Gamma))
            
        if config_new['summary_statistics']['summary_statistic'] == '3d_count':
            mask0 = log10mWL != None
            threed_hist = np.zeros([len(self.richness_edges)-1, len(self.redshift_edges)-1, len(self.log10mWL_edges)-1])
            twod_bins = [self.richness_edges, self.redshift_edges,]
            for i in range(len(self.log10mWL_edges)-1):
                mask_mass = (log10mWL[mask0] > self.log10mWL_edges[i]) * (log10mWL[mask0] < self.log10mWL_edges[i+1])
                threed_hist[:,:,i] = np.histogram2d(richness[mask0][mask_mass], z_obs[mask0][mask_mass],bins=twod_bins)[0]
            return threed_hist

















            