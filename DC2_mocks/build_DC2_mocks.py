import numpy as np
import jaxlib, jax
import jax.numpy as jnp
import equinox, optax
import matplotlib.pyplot as plt
import sys
import pyccl as ccl
import classy
import PySSC
import pyccl
from astropy.table import Table
from pathlib import Path
# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from modules.simulation import UniverseSimulator
import configparser

import clmm
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
from clmm.support import mock_data as mock

import configparser

def ccl_cosmo(config_new):
    Omega_m = float(config_new['parameters']['Omega_m'])
    Omega_b = float(config_new['parameters']['Omega_b'])
    sigma8 = float(config_new['parameters']['sigma8'])
    h = float(config_new['parameters']['h'])
    ns = float(config_new['parameters']['ns'])
    w0 = float(config_new['parameters']['w0'])
    wa = float(config_new['parameters']['wa'])
    cosmo_ccl_object = ccl.Cosmology( Omega_c = Omega_m - Omega_b, Omega_b = Omega_b, h = h , sigma8 = sigma8, n_s= ns,
                          transfer_function='boltzmann_class', matter_power_spectrum='linear')

    cosmo_clmm_object = Cosmology(H0=100*h, Omega_dm0=Omega_m - Omega_b, Omega_b0=Omega_b, Omega_k0=0.0)

    cosmo_clmm_object.be_cosmo = cosmo_ccl_object

    return cosmo_ccl_object, cosmo_clmm_object

def generate_profile(log10m, z, c, cosmo):
    
    noisy_data_z = mock.generate_galaxy_catalog(
    10**log10m, z, c,
    cosmo, "chang13", zsrc_min=z+0.2,
    shapenoise=0.25,
    massdef="critical", photoz_sigma_unscaled=1e-8,
    ngal_density=25, cluster_ra=0,cluster_dec=0,
    field_size=7 #Mpc
    ,)
    cl = GalaxyCluster('id', 0, 0, z, noisy_data_z, )
    cl.compute_tangential_and_cross_components(add=True, cosmo=cosmo, is_deltasigma=True)
    new_profiles = cl.make_radial_profile("Mpc", bins=new_bins, cosmo=cosmo)

    return new_profiles['radius'], new_profiles['gt'], new_profiles['W_l']

default_config_capish = configparser.ConfigParser()
default_config_capish = configparser.ConfigParser()
default_config_capish.read('capish_config_DC2like.ini')

cosmo_ccl_object, cosmo_clmm_object = ccl_cosmo(default_config_capish)

sim = UniverseSimulator(default_config_path = None , default_config = default_config_capish, 
                        variable_params_names = ['Omega_m','sigma8'])

log10m_halo, z_true, richness, log10mWL, z_obs = sim.run_simulation_halo_and_cluster_catalogue([0.2648, 0.8])

mask_cluster_catalog = (z_obs >= 0.2)*(z_obs <= 0.8)*(richness>=20)*(richness<=200)

new_bins = make_bins(1.1, 3.7, nbins=7, method="evenlog10width")

radius_list = []
DS_list = []
Wl_list = []
#indexes = np.random.choice(np.arange(len(log10m_halo))[mask_cluster_catalog], 300, replace=False)
indexes = np.arange(len(log10m_halo))[mask_cluster_catalog]
for index in indexes:
    np.random.seed(index)
    r, DS, Wl = generate_profile(log10m_halo[index], z_true[index], 3.8, cosmo_clmm_object)
    radius_list.append(r)
    DS_list.append(DS)
    Wl_list.append(Wl)

Summary_table = Table()

Summary_table['binned_radius_Mpc'] = np.array(radius_list)
Summary_table['binned_profiles'] = np.array(DS_list)
Summary_table['binned_weights'] = np.array(Wl_list)
Summary_table['cluster_redshift'] = z_obs[indexes]
Summary_table['cluster_richness'] = richness[indexes]

Summary_table.write('mock_DC2like_data.fits',overwrite=True)