import numpy as np

def photometric_redshift(z_true, params_photoz):

    sigma_pz0 = params_photoz
    z_obs = z_true + np.random.randn(len(z_true)) * sigma_pz0 * (z_true + 1)
    return z_obs