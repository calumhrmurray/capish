import numpy as np

richness_bins = np.array( [ 20 , 30 , 45 , 60 , 2400 ])
redshift_bins = np.array( [ 0.2 , 0.35 , 0.5 , 0.65 ])

richness_cents = [ 25 , 37.5 , 52.5 , 100 ]
redshift_cents = 0.5 * (redshift_bins[1:] + redshift_bins[:-1])

# Std for Mean mass results, Table II DES Y1 cluster abundance results
mwl_std_0 = np.array( [ 0.032 + 0.045 , 0.031 + 0.051 , 0.044 + 0.050 , 0.038 + 0.052 ] )
mwl_std_1 = np.array( [ 0.033 + 0.056 , 0.031 + 0.061 , 0.044 + 0.065 , 0.038 + 0.052 ] )
mwl_std_2 = np.array( [ 0.048 + 0.072 , 0.041 + 0.086 , 0.056 + 0.068 , 0.061 + 0.069 ] )

mwl_std = np.array( [ mwl_std_0, mwl_std_1, mwl_std_2 ] ).T


def counts_and_mean_mass( cluster_catalogue , 
                          richness_bins = richness_bins , 
                          redshift_bins = redshift_bins , 
                          mwl_std = mwl_std,
                          exp = 2/3 ):
    
    richness = cluster_catalogue['richness']
    redshift = cluster_catalogue['redshift']
    log10M_wl = cluster_catalogue['log10_mwl']

    # Compute the 2D histogram for cluster counts
    observed_cluster_abundance, _, _ = np.histogram2d(
        richness, 
        redshift, 
        bins = [ richness_bins, redshift_bins ]
    )


    # Compute the 2D histogram for the sum of log10M_wl
    sum_log10M_wl, _, _ = np.histogram2d(  richness, 
                                            redshift, 
                                            bins = [ richness_bins, redshift_bins], 
                                            weights = ( 10**log10M_wl )**(1/exp)
    )

    # Calculate mean log10M_wl in each bin (avoid division by zero)
    # we cannot add inverse variance weights here, since we pretend not to know the mass
    # is this fully consistent?
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_log10M_wl = np.where( observed_cluster_abundance > 0, 
                                    sum_log10M_wl / observed_cluster_abundance, 
                                    1 )
        mean_log10M_wl = np.log10( mean_log10M_wl**(exp) )
        
    # add measurement and systematic uncertainties
    mean_log10M_wl = mean_log10M_wl + np.random.normal( loc = 0.0 , scale = mwl_std )
        
    return np.vstack( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten() 