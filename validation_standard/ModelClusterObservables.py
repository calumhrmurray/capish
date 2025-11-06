import sys
from scipy.stats import norm, poisson
import numpy as np
import pyccl as ccl
import configparser
#import _ModelClusterAbundance as _cluster_abundance
import ModelRichnessMassRelation as rm_relation
path = '/pbs/throng/lsst/users/cpayerne/capish/modules/'
sys.path.append(path)
import halo.halo_catalogue as halocat_mod
import halo._halo_abundance as halocount_mod
import cluster._completeness as comp
import cluster._purity as pur
from scipy.integrate import simps

def binning(edges): return [[edges[i],edges[i+1]] for i in range(len(edges)-1)]

def reshape_axis(axis, bounds):
    
    index = np.arange(len(axis))
    mask = (axis > bounds[0])*(axis < bounds[1])
    index_cut = index[mask]
    axis_cut = axis[mask]
    axis_cut[0], axis_cut[-1] = bounds
    return index_cut, axis_cut

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

def reshape_integrand(integrand_count, index_richness_grid_cut, index_z_grid_cut):
        r"""reshape integrand_count on selected indexes for the richness axis and redshift axis
        Attributes:
        -----------
        integrand_count: array
            integrand over the mass, richness and redshift axis
        index_richness_grid_cut: array
            indexes on the richness axis
        index_richness_grid_cut: array
            indexes on the redshift axis
        Returns:
        --------
        integrand_cut: array
            reshaped integrand
        """
        integrand_cut = integrand_count[index_richness_grid_cut,:,:]
        integrand_cut = integrand_cut[:,:,index_z_grid_cut]
        return integrand_cut

class UniversePrediction:
    
    def __init__(self, default_config_path = None , default_config = None):

        if default_config == None:
            default_config = configparser.ConfigParser()
            default_config.read(default_config_path)

        print(default_config)
        massdef = halocat_mod.get_massdef_from_config(default_config['halo_catalogue'])
        hmf = halocat_mod.get_massfunc_from_config(default_config['halo_catalogue'])
        halobias_fct = halocat_mod.get_bias_from_config(default_config['halo_catalogue'])
       
        parameters = default_config['parameters']
        rm_param_names = ['M_min', 'alpha_lambda', 'beta_lambda', 'gamma_lambda', 'sigma_lambda']
        theta_rm = [float(parameters[n]) for n in rm_param_names]
        which = default_config['cluster_catalogue.mass_observable_relation']['which_relation']
        RM_count_and_mass = rm_relation.Richness_mass_relation()
        RM_count_and_mass.select(which = which)

        logm_grid = np.linspace( float( default_config['halo_catalogue']['log10m_min']),
                              float( default_config['halo_catalogue']['log10m_max']), 300 )

        #redshift grid
        z_grid = np.linspace( float( default_config['halo_catalogue']['z_min'] ),
                            float( default_config['halo_catalogue']['z_max'] ), 302 )
        richness_grid = np.logspace(np.log10(20), np.log10(200), 301)

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
        self.ClusterAbundanceObject = ClusterAbundanceMath(HaloAbundanceObject=self.HaloAbundanceObject, MoRObject=RM_count_and_mass)


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

        Z_bin = binning(self.redshift_edges)
        Richness_bin = binning(self.richness_edges)

        self.bins = {'redshift_bins':Z_bin, 'richness_bins': Richness_bin}
        self.grids = {'logm_grid': logm_grid, 'z_grid': z_grid, 'richness_grid':richness_grid}

        count_modelling = {'dNdzdlogMdOmega':None,'richness_mass_relation':None, 'completeness':None, 'purity':None }

        theta_purity = [float(k) for k in default_config['cluster_catalogue']['params_purity'].split(', ')]
        theta_completeness = [float(k) for k in default_config['cluster_catalogue']['params_completeness'].split(', ')]
    
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

    def model_mass(self, params_new, compute_new, adds_new, gamma = 0.7, add_WL_weight = False):
        count_modelling_new = self.ClusterAbundanceObject.recompute_count_modelling(self.count_modelling_defaut, grids = self.grids, compute = compute_new, params=params_new)
        integrand_count = self.ClusterAbundanceObject.define_count_integrand(count_modelling_new, adds_new)
        Omega = (self.HaloAbundanceObject.sky_area/(4*np.pi))*(4*np.pi)
        if add_WL_weight == False: 
            Nth = Omega * self.ClusterAbundanceObject.Cluster_SurfaceDensity_ProxyZ(self.bins, integrand_count = integrand_count, grids = self.grids)
            NthMwl = Omega * self.ClusterAbundanceObject.Cluster_dNd0mega_Mass_ProxyZ(self.bins, integrand_count = integrand_count, grids = self.grids, gamma=gamma)
            return NthMwl, Nth
        if add_WL_weight == True: 
            z_grid = self.grids['z_grid']
            WLz = lensing_weights(self.params_default['CCL_cosmology'], z_grid) * 1e32
            WLz_expanded = WLz[np.newaxis, np.newaxis, :]
            integrand_count_w = integrand_count * WLz_expanded
            Nth_w = Omega * self.ClusterAbundanceObject.Cluster_SurfaceDensity_ProxyZ(self.bins, integrand_count = integrand_count_w, grids = self.grids)
            NthMwl_w = Omega * self.ClusterAbundanceObject.Cluster_dNd0mega_Mass_ProxyZ(self.bins, integrand_count = integrand_count_w , grids = self.grids, gamma=gamma)
            return NthMwl_w, Nth_w
            
    def model_bias(self, params_new, compute_new, adds_new):
        count_modelling_new = self.ClusterAbundanceObject.recompute_count_modelling(self.count_modelling_defaut, grids = self.grids, compute = compute_new, params=params_new)
        integrand_count = self.ClusterAbundanceObject.define_count_integrand(count_modelling_new, adds_new)
        Nb = self.ClusterAbundanceObject.Cluster_NHaloBias_ProxyZ(self.bins, integrand_count = integrand_count, halo_bias = count_modelling_new['halo_bias'], grids = self.grids)
        return (self.HaloAbundanceObject.sky_area/(4*np.pi))*(4*np.pi) * Nb
        
        

#does the math calculation for 
class ClusterAbundanceMath():

    def __init__(self, HaloAbundanceObject=None, MoRObject = None):
        #predict the halo mass function
        self.HaloAbundanceObject = HaloAbundanceObject
        #predict the MoR
        self.mass_richness_relation = MoRObject
        return None
    
    def compute_dNdzdlogMdOmega_grid(self, logm_grid, z_grid, cosmo):
        r"""
        Attributes:
        -----------
        logm_grid : array
            log10M tabulated axis
        z_grid : array
            redshift tabulated axis
        cosmo : CCL cosmology
            cosmology
        hmd; CCL hmd object
            halo definition
        Returns:
        --------
        dN_dzdlogMdOmega : array
            tabulated dzdlogMdOmega
        """
        dNdzdlogMdOmega_grid = np.zeros([len(logm_grid), len(z_grid)])
        for i, z in enumerate(z_grid):
            dNdzdlogMdOmega_grid[:,i] = self.HaloAbundanceObject.dndlog10M(logm_grid ,z, cosmo) * self.HaloAbundanceObject.dVdzdOmega(z, cosmo)
        return dNdzdlogMdOmega_grid
    
    def compute_dNdzdlogMdOmega_log_slope_grid(self, logm_grid, z_grid, cosmo):
        r"""
        Attributes:
        -----------
        logm_grid : array
            log10M tabulated axis
        z_grid : array
            redshift tabulated axis
        cosmo : CCL cosmology
            cosmology
        hmd; CCL hmd object
            halo definition
        Returns:
        --------
        dN_dzdlogMdOmega : array
            tabulated dzdlogMdOmega
        """
        dNdzdlogMdOmega_log_slope_grid = np.zeros([len(logm_grid), len(z_grid)])
        for i, z in enumerate(z_grid):
            ln = np.log10(self.HaloAbundanceObject.dndlog10M(logm_grid - 0.001 ,z, cosmo) * self.HaloAbundanceObject.dVdzdOmega(z, cosmo))
            ln_dm = np.log10(self.HaloAbundanceObject.dndlog10M(logm_grid + 0.001 ,z, cosmo) * self.HaloAbundanceObject.dVdzdOmega(z, cosmo))
            dNdzdlogMdOmega_log_slope_grid[:,i] = -(ln_dm - ln)/(2*0.001)
        return dNdzdlogMdOmega_log_slope_grid
    
    def compute_halo_bias_grid(self, logm_grid, z_grid, cosmo):
        r"""
        Attributes:
        -----------
        logm_grid : array
            log10M tabulated axis
        z_grid : array
            redshift tabulated axis
        cosmo : CCL cosmology
            cosmology
        Returns:
        --------
        dN_dzdlogMdOmega : array
            tabulated dzdlogMdOmega
        """
        halo_bias = np.zeros([len(logm_grid), len(z_grid)])
        for i, z in enumerate(z_grid):
            halo_bias[:,i] = self.HaloAbundanceObject.CCLBias.__call__(cosmo, 10**logm_grid, 1/(1+z))
        return halo_bias
    
    
    def compute_purity_grid(self, richness_grid, z_grid, theta_purity):
        r"""
        based on https://arxiv.org/pdf/1611.05468.pdf
        Attributes:
        -----------
        richness_grid : array
            richness tabulated axis
        z_grid : array
            redshift tabulated axis
        theta_purity : array
            parameters of purity
        Returns:
        --------
        purity_grid : array
            tabulated purity
        """
        R, Z = np.meshgrid(richness_grid, z_grid)
        R_flat, Z_flat = R.T.flatten(), Z.T.flatten()
        purity_flat = pur.purity(R_flat, Z_flat, params=theta_purity)
        purity_grid = np.reshape(purity_flat, [len(richness_grid), len(z_grid)])
        return purity_grid
    
    def compute_completeness_grid(self, logm_grid, z_grid, theta_completeness):
        r"""
        based on https://arxiv.org/pdf/1611.05468.pdf
        Attributes:
        -----------
        logm_grid : array
            log10M tabulated axis
        z_grid : array
            redshift tabulated axis
        theta_completeness : array
            parameters of purity
        Returns:
        --------
        completeness_grid : array
            tabulated completeness
        """
        Logm, Z = np.meshgrid(logm_grid, z_grid)
        Logm_flat, Z_flat = Logm.T.flatten(), Z.T.flatten()
        completeness_flat = comp.completeness(Logm_flat, Z_flat, params=theta_completeness)
        completeness_grid = np.reshape(completeness_flat, [len(logm_grid), len(z_grid)])
        return completeness_grid
    
    def compute_richness_mass_relation_grid(self, richness_grid, logm_grid, z_grid, theta_rm):
        r"""
        based on https://arxiv.org/pdf/1904.07524.pdf
        Attributes:
        -----------
        richness_grid : array
            richness tabulated axis
        logm_grid : array
            log10M tabulated axis
        z_grid : array
            redshift tabulated axis
        theta_rm : array
            parameters of purity
        Returns:
        --------
        rm_grid : array
            tabulated richness-mass relation
        """
        rm_relation_grid = np.zeros([len(richness_grid), len(logm_grid), len(z_grid)])
        richness_tab = np.zeros([len(richness_grid), len(logm_grid), len(z_grid)])
        logm_tab = np.zeros([len(richness_grid), len(logm_grid), len(z_grid)])
        z_tab = np.zeros([len(richness_grid), len(logm_grid), len(z_grid)])
        for i, richness in enumerate(richness_grid): richness_tab[i,:,:] = richness
        for i, logm in enumerate(logm_grid): logm_tab[:,i,:] = logm
        for i, z in enumerate(z_grid): z_tab[:,:,i] = z
        mu = self.mass_richness_relation.proxy_mu_f(logm_tab, z_tab, theta_rm)
        sigma = self.mass_richness_relation.proxy_sigma_f(logm_tab, z_tab, theta_rm)
        pdf = self.mass_richness_relation.pdf_richness_mass_relation(richness_tab, logm_tab, z_tab, theta_rm)
        return pdf, mu, sigma
    
    def recompute_count_modelling(self, count_modelling, compute = None, grids = None, params = None):
        r"""
        recompute the tabulated maps for cluster abundance integrand
        Attributes:
        -----------
        count_modelling: dict
            dictionary of tabulated maps of hmf, richness-mass relation, purity, completeness
        compute: dict
            dictionnary of boolean to choose to compute
        grids: dict
            dictionary of tabulated axis for the mass, richness, redshift
        params: dict
            dictionary of params (cosmology, M-lambda relation, purity and completeness
        Returns:
        --------
        count_modelling: dict
            dictionary of tabulated (recomputed) maps of hmf, richness-mass relation, purity, completeness
        """
        
        richness_grid, logm_grid, z_grid = grids['richness_grid'], grids['logm_grid'], grids['z_grid']
        shape_integrand = [len(richness_grid), len(logm_grid), len(z_grid)]
        
        if compute['compute_purity']:
            purity_ = self.compute_purity_grid(grids['richness_grid'], grids['z_grid'], params['params_purity'])
            purity_new = np.zeros(shape_integrand)
            for i in range(shape_integrand[1]):
                purity_new[:,i,:] = purity_
            count_modelling['purity'] = purity_new
                
        if compute['compute_completeness']:
            completeness_ = self.compute_completeness_grid(grids['logm_grid'], grids['z_grid'], params['params_completeness'])
            completeness_new = np.zeros(shape_integrand)
            for i in range(shape_integrand[0]):
                completeness_new[i,:,:] = completeness_
            count_modelling['completeness'] = completeness_new
         
        if compute['compute_richness_mass_relation']:
            pdf_map, mu_map, sigma_map = self.compute_richness_mass_relation_grid(grids['richness_grid'], 
                                                                             grids['logm_grid'], 
                                                                             grids['z_grid'], 
                                                                             params['params_richness_mass_relation'])
            count_modelling['richness_mass_relation'] = pdf_map
            count_modelling['richness_mass_relation - mean'] = mu_map
            count_modelling['richness_mass_relation - sigma'] = sigma_map
    
        if compute['compute_dNdzdlogMdOmega']:
            dNdzdlogMdOmega_ = self.compute_dNdzdlogMdOmega_grid(grids['logm_grid'], grids['z_grid'], 
                                                            params['CCL_cosmology'])
            dNdzdlogMdOmega_new = np.zeros(shape_integrand)
            for i in range(shape_integrand[0]):
                dNdzdlogMdOmega_new[i,:,:] = dNdzdlogMdOmega_
            count_modelling['dNdzdlogMdOmega'] = dNdzdlogMdOmega_new
            
        if compute['compute_dNdzdlogMdOmega_log_slope']:
            dNdzdlogMdOmega_log_slope_ = self.compute_dNdzdlogMdOmega_log_slope_grid(grids['logm_grid'], grids['z_grid'], 
                                                            params['CCL_cosmology'])
            dNdzdlogMdOmega_log_slope_new = np.zeros(shape_integrand)
            for i in range(shape_integrand[0]):
                dNdzdlogMdOmega_log_slope_new[i,:,:] = dNdzdlogMdOmega_log_slope_
            count_modelling['dNdzdlogMdOmega_log_slope'] = dNdzdlogMdOmega_log_slope_new
            
        if compute['compute_halo_bias']:
            count_modelling['halo_bias'] = self.compute_halo_bias_grid(grids['logm_grid'], grids['z_grid'], 
                                                            params['CCL_cosmology'])
            
        
        return count_modelling
    
    def define_count_integrand(self, count_modelling, adds, core = 'dNdzdlogMdOmega'):
        r"""
        define count integrand with the option of considering purity and/or completeness
        Attributes:
        ----------
        count_modelling: dict
            dictionnary of tabulated hmf, purity, completeness, richness-mass relation
        adds: dict
            dictionary of booleans, choose if purity and/or completeness are considered
        Returns:
        --------
        integrand: array
            integrand on the mass, richness and redshift axis
        """
        dNdzdlogMdOmega = count_modelling[core]
        richness_mass_relation = count_modelling['richness_mass_relation']
        purity = count_modelling['purity']
        completeness = count_modelling['completeness']
        
        if adds['add_purity']:
            if adds['add_completeness']:
                integrand = dNdzdlogMdOmega * richness_mass_relation * completeness / purity
            else:
                integrand = dNdzdlogMdOmega * richness_mass_relation / purity
        else:
            if adds['add_completeness']:
                integrand = dNdzdlogMdOmega * richness_mass_relation * completeness
            else:
                integrand = dNdzdlogMdOmega * richness_mass_relation
        return integrand
    
    def Cluster_SurfaceDensity_ProxyZ(self, bins, integrand_count = None, grids = None): 
     
        r"""
        Attributes:
        -----------
        bins: dict
            dictionnary of redshift and richness bins
        integrand_count: array
            tabulated integrand, to be integrated over the amss, richness and redshift axis
        grids: dict
            dictionnary of tabulated arrays of the mass, richness, redshift axis
        Returns:
        --------
        dNdOmega: array
            angular surface density of clusters in bins of redshift and mass
        """
        richness_grid, logm_grid, z_grid = grids['richness_grid'], grids['logm_grid'], grids['z_grid']
        z_bins, richness_bins = bins['redshift_bins'], bins['richness_bins']
        
        dNdOmega = np.zeros([len(richness_bins), len(z_bins)])
        for i, richness_bin in enumerate(richness_bins):
            #resize richness-axis
            index_richness_grid_cut, richness_grid_cut = reshape_axis(richness_grid, richness_bin)
            for j, z_bin in enumerate(z_bins):
                #resize redshift-axis
                index_z_grid_cut, z_grid_cut = reshape_axis(z_grid, z_bin)
                integrand_cut = reshape_integrand(integrand_count, index_richness_grid_cut, index_z_grid_cut)
                integral = simps(simps(simps(integrand_cut, 
                                             richness_grid_cut, axis=0), 
                                             logm_grid, axis=0), 
                                             z_grid_cut, axis=0)
                dNdOmega[i,j] = integral
        return dNdOmega

    def Cluster_dNd0mega_Mass_ProxyZ(self, bins, integrand_count = None, grids = None, gamma = 1): 
    
        richness_grid, logm_grid, z_grid = grids['richness_grid'], grids['logm_grid'], grids['z_grid']
        z_bins, richness_bins = bins['redshift_bins'], bins['richness_bins']
                
        Nstacked_masses = np.zeros([len(richness_bins), len(z_bins)])
    
        for i, richness_bin in enumerate(richness_bins):
            #resize richness-axis
            index_richness_grid_cut, richness_grid_cut = reshape_axis(richness_grid, richness_bin)
            for j, z_bin in enumerate(z_bins):
                #resize redshift-axis
                index_z_grid_cut, z_grid_cut = reshape_axis(z_grid, z_bin)
                m_grid_mat = 10 ** np.tile(logm_grid, (len(z_grid_cut), 1)).T
                integrand_cut = reshape_integrand(integrand_count, index_richness_grid_cut, index_z_grid_cut)
                m_grid_mat_int = m_grid_mat ** gamma
                    
                Nstacked_masses[i,j] = simps(simps(simps(integrand_cut, 
                                             richness_grid_cut, axis=0) * m_grid_mat_int,
                                             logm_grid, axis=0), 
                                             z_grid_cut, axis=0)
        return Nstacked_masses
    
    def Cluster_NHaloBias_ProxyZ(self, bins, integrand_count = None, halo_bias = None, grids = None, cosmo = None):
    
        richness_grid, logm_grid, z_grid = grids['richness_grid'], grids['logm_grid'], grids['z_grid']
        z_bins, richness_bins = bins['redshift_bins'], bins['richness_bins']
        bias_dNdOmega = np.zeros([len(richness_bins), len(z_bins)])
        for i, richness_bin in enumerate(richness_bins):
            #resize richness-axis
            index_richness_grid_cut, richness_grid_cut = reshape_axis(richness_grid, richness_bin)
            for j, z_bin in enumerate(z_bins):
                #resize redshift-axis
                index_z_grid_cut, z_grid_cut = reshape_axis(z_grid, z_bin)
                integrand_cut = reshape_integrand(integrand_count, index_richness_grid_cut, index_z_grid_cut)
                halo_bias_cut = halo_bias[:,index_z_grid_cut]
                integral = simps(simps(simps(integrand_cut, 
                                             richness_grid_cut, axis=0) * halo_bias_cut, 
                                             logm_grid, axis=0), 
                                             z_grid_cut, axis=0)
                bias_dNdOmega[i,j] = integral
        return bias_dNdOmega