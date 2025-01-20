import numpy as np
log10m0, z0 = np.log10(10**14.3), .5
proxy_mu0, proxy_muz, proxy_mulog10m =  3.2, 0.0, 2.2
proxy_sigma0, proxy_sigmaz, proxy_sigmalog10m =  0.5, 0, 0
sigma_wl_log10mass = 0.25/np.log(10) #infinite source density 
which_model = 'log_normal_poisson_scatter'