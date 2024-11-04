import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import covariance as covar
import abundance as cl_count
import pyccl as ccl

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

print('[define input cosmology]')
Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711
cosmo = ccl.Cosmology(Omega_c = Omega_c_true + Omega_b_true - 0.048254, Omega_b = 0.048254, 
                          h = 0.6777, sigma8 = sigma8_true, n_s=0.96)

print('[define mass & redshift range of the halo catalog, and sky coverage]')
z_range = [0.2, 1]
logm_range = [14, 16]
fsky = 0.25
nzbins = 3
nmbins = 5
z_corner = np.linspace(z_range[0], z_range[1], nzbins+1)
log10m_corner = np.linspace(logm_range[0], logm_range[1], nmbins+1)
Z_bin = [[z_corner[i], z_corner[i+1]] for i in range(len(z_corner)-1)]
LogMass_bin = [[log10m_corner[i], log10m_corner[i+1]] for i in range(len(log10m_corner)-1)]

print('[define mass & redshift grids for numerical integration]')
zmin, zmax=0.2, 1.2
logmmin, logmmax=14, 16
z_grid = np.linspace(zmin, zmax, 100)
logm_grid = np.linspace(logmmin, logmmax, 1001)

print('[initiate cluster count object from class ClusterAbundance()]')
clc = cl_count.ClusterAbundance()
clc.sky_area = (fsky)*4*np.pi
clc.f_sky = fsky
massdef = ccl.halos.massdef.MassDef('vir', 'critical')
hmd = ccl.halos.hmfunc.MassFuncDespali16(mass_def=massdef)
halobias = ccl.halos.hbias.HaloBiasTinker10(mass_def= massdef, mass_def_strict=True)
clc.set_cosmology(cosmo = cosmo, hmd = hmd, massdef = massdef)
clc.sky_area = fsky * 4 * np.pi
clc.f_sky = clc.sky_area/(4*np.pi)
clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
clc.compute_halo_bias_grid_MZ(z_grid = z_corner, logm_grid = log10m_corner, halobiais = halobias)
print('[compute count prediction in mass & redshift bins]')
N_pred = clc.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')
print('[compute prediction of average bias * N in mass & redshift bins]')
NHalo_bias_pred = clc.Nhalo_bias_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')

print('[initiate covariance object from class Covariance_matrix()]')
Covariance = covar.Covariance_matrix()
print('[compute S_ij matrix]')
Sij_fullsky = Covariance.matter_fluctuation_amplitude_fullsky(Z_bin, cosmo = cosmo, approx = True)
Sij_partialsky = Sij_fullsky/clc.f_sky
print('[compute b_i * b_j * N_i * N_j * S_ij matrix]')
NNSbb = Covariance.sample_covariance_full_sky(Z_bin, LogMass_bin, 
                                                  NHalo_bias_pred, 
                                                  Sij_partialsky)
print('[compute total covariance matrix]')
Cov = NNSbb + np.diag(N_pred.flatten())
