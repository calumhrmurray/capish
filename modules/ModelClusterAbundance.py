import sys
from scipy.stats import norm, poisson
import numpy as np
import pyccl as ccl
import configparser
import _ModelClusterAbundance as _cluster_abundance
import ModelRichnessMassRelation as rm_relation
sys.path.append('../')
import modules.halo.halo_catalogue as halocat_mod
import modules.halo._halo_abundance as halocount_mod

class UniversePrediction:
    
    def __init__(self, default_config_path = None , default_config = None):


        default_config = configparser.ConfigParser()
        default_config.read(default_config_path)

        print(default_config)
        massdef = halocat_mod.get_massdef_from_config(default_config['halo_catalogue'])
        hmf = halocat_mod.get_massfunc_from_config(default_config['halo_catalogue'])
        halobias_fct = halocat_mod.get_bias_from_config(default_config['halo_catalogue'])
       
        parameters = default_config['parameters']
        pivot_obs_z0 = float(parameters['pivot_obs_z0'])
        pivot_obs_log10m0 = float(parameters['pivot_obs_log10m0'])
        params_observable_mean = [float(parameters['params_mean_obs_mu0']), float(parameters['params_mean_obs_muz']), float(parameters['params_mean_obs_mulog10m'])]
        params_observable_stdd = [float(parameters['params_stdd_obs_mu0']), float(parameters['params_stdd_obs_muz']), float(parameters['params_stdd_obs_mulog10m'])]
        theta_rm = [pivot_obs_log10m0, pivot_obs_z0] + params_observable_mean + params_observable_stdd
        which = default_config['cluster_catalogue.mass_observable_relation']['which_relation']
        RM_count_and_mass = rm_relation.Richness_mass_relation()
        RM_count_and_mass.select(which = which)

        logm_grid = np.linspace( float( default_config['halo_catalogue']['log10m_min']),
                              float( default_config['halo_catalogue']['log10m_max']),
                              300 )

        #redshift grid
        z_grid = np.linspace( float( default_config['halo_catalogue']['z_min'] ),
                            float( default_config['halo_catalogue']['z_max'] ),
                            300 )
        richness_grid = np.logspace(np.log10(20), np.log10(200), 310)

        Omega_m = float(default_config['parameters']['Omega_m'])
        Omega_b = float(default_config['parameters']['Omega_b'])
        sigma8 = float(default_config['parameters']['sigma8'])
        h = float(default_config['parameters']['h'])
        ns = float(default_config['parameters']['ns'])
        w0 = float(default_config['parameters']['w0'])
        wa = float(default_config['parameters']['wa'])

        sky_area = float( default_config['halo_catalogue']['sky_area'] )
        fsky = sky_area/(4*np.pi) #%

        cosmo_fid = ccl.Cosmology( Omega_c = Omega_m - Omega_b, Omega_b = Omega_b, h = h , sigma8 = sigma8, n_s= ns)

        self.HaloAbundanceObject = halocount_mod.HaloAbundance(CCLCosmologyObject = cosmo_fid, CCLHmf = hmf, CCLBias = halobias_fct, sky_area = sky_area)
        self.ClusterAbundanceObject = _cluster_abundance.ClusterAbundance(HaloAbundanceObject=self.HaloAbundanceObject, MoRObject=RM_count_and_mass)


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

        
        def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
        Z_bin = binning(self.redshift_edges)
        Richness_bin = binning(self.richness_edges)

        self.bins = {'redshift_bins':Z_bin, 'richness_bins': Richness_bin}
        self.grids = {'logm_grid': logm_grid, 'z_grid': z_grid, 'richness_grid':richness_grid}

        count_modelling = {'dNdzdlogMdOmega':None,'richness_mass_relation':None, 'completeness':None, 'purity':None }

        a_nc, b_nc, a_rc, b_rc = np.log(10)*0.8612, np.log(10)*0.3527, 2.2183, -0.6592
        theta_purity = [a_nc, b_nc, a_rc, b_rc]
        a_nc, b_nc, a_mc, b_mc = 1.1321, 0.7751, 13.31, 0.2025
        theta_completeness = [a_nc, b_nc, a_mc, b_mc]
    
        self.params_default = {'params_richness_mass_relation': theta_rm,
                             'CCL_cosmology': cosmo_fid, 
                               'params_completeness': theta_completeness,
                               'params_purity' : theta_purity}
        compute_default = {'compute_dNdzdlogMdOmega':True,'compute_richness_mass_relation':True, 
                               'compute_completeness':True, 'compute_purity':True ,'compute_halo_bias':True,
                             'compute_dNdzdlogMdOmega_log_slope': False}

        self.count_modelling_defaut = self.ClusterAbundanceObject.recompute_count_modelling(compute_default, grids = self.grids, compute = compute_default, params = self.params_default)
        
    def model_count(self, params_new, compute_new, adds_new):
        count_modelling_new = self.ClusterAbundanceObject.recompute_count_modelling(self.count_modelling_defaut, grids = self.grids, compute = compute_new, params=params_new)
        integrand_count = self.ClusterAbundanceObject.define_count_integrand(count_modelling_new, adds_new)
        Omega = (self.HaloAbundanceObject.sky_area/(4*np.pi))*(4*np.pi)
        Nth = Omega * self.ClusterAbundanceObject.Cluster_SurfaceDensity_ProxyZ(self.bins, integrand_count = integrand_count, grids = self.grids)
        return Nth
