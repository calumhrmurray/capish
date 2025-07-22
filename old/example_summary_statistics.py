def three_dim_counts(self, richness, log10M_wl, z_clusters ):
    """
    Calculate the number of clusters in bins of cluster richness, redshift, and weak lensing mass.

    Parameters:
    - richness: array-like, cluster richness values
    - log10M_wl: array-like, log10 of weak lensing mass values
    - z_clusters: array-like, redshift values of the clusters
    - richness_bins: array-like, edges of the bins for richness
    - redshift_bins: array-like, edges of the bins for redshift
    - mass_bins: array-like, edges of the bins for weak lensing mass (log10)

    Returns:
    - counts: a 3D array of shape (len(richness_bins)-1, len(redshift_bins)-1, len(mass_bins)-1)
    """
    
    
    # Calculate histogram counts in 3D bins
    # the lensing mass counts are inverse variance weighted
    counts, edges = np.histogramdd(
        np.column_stack([ richness, 
                            z_clusters, 
                            log10M_wl ]),
        bins=[ self.small_richness_bins, 
                self.small_redshift_bins, 
                self.small_log10Mwl_bins ]
    )

    return counts.flatten()

def stacked_counts( self , richness, log10M_wl, redshift , exp =  1):
    """
    Calculate the number of clusters in bins of cluster richness and redshift,
    and calculate the mean cluster weak-lensing mass in these bins.

    Parameters:
    richness (array-like): Array of richness values.
    log10M_wl (array-like): Array of weak-lensing mass values (log10 scale).
    redshift (array-like): Array of redshift values.
    richness_bins (array-like): Bin edges for richness.
    redshift_bins (array-like): Bin edges for redshift.

    Returns:
    observed_cluster_abundance (2D array): Number of clusters in each bin.
    mean_log10M_wl (2D array): Mean log10 weak-lensing mass in each bin.
    """
    # Compute the 2D histogram for cluster counts
    observed_cluster_abundance, _, _ = np.histogram2d(
        richness, 
        redshift, 
        bins = [ self.richness_bins, self.redshift_bins ]
    )


    # Compute the 2D histogram for the sum of log10M_wl
    sum_log10M_wl, _, _ = np.histogram2d(  richness, 
                                            redshift, 
                                            bins = [ self.richness_bins, self.redshift_bins], 
                                            weights = ( log10M_wl )
    )

    # Calculate mean log10M_wl in each bin (avoid division by zero)
    # we cannot add inverse variance weights here, since we pretend not to know the mass
    # is this fully consistent?
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_log10M_wl = np.where( observed_cluster_abundance > 0, 
                                    sum_log10M_wl / observed_cluster_abundance, 
                                    1 )
        
    # add measurement and systematic uncertainties
    if self.include_mwl_measurement_errors:
        mean_log10M_wl = mean_log10M_wl + np.random.normal( loc = 0.0 , scale = self.mwl_std )
        
    return np.vstack( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten()  
        
def des_stacked_counts( self , richness, log10M_wl, redshift , exp =  1):
    """
    Calculate the number of clusters in bins of cluster richness and redshift,
    and calculate the mean cluster weak-lensing mass in these bins.

    Parameters:
    richness (array-like): Array of richness values.
    log10M_wl (array-like): Array of weak-lensing mass values (log10 scale).
    redshift (array-like): Array of redshift values.
    richness_bins (array-like): Bin edges for richness.
    redshift_bins (array-like): Bin edges for redshift.

    Returns:
    observed_cluster_abundance (2D array): Number of clusters in each bin.
    mean_log10M_wl (2D array): Mean log10 weak-lensing mass in each bin.
    """
    # Compute the 2D histogram for cluster counts
    observed_cluster_abundance, _, _ = np.histogram2d(
        richness, 
        redshift, 
        bins = [ self.richness_bins, self.redshift_bins ]
    )


    # Compute the 2D histogram for the sum of log10M_wl
    sum_log10M_wl, _, _ = np.histogram2d(  richness, 
                                            redshift, 
                                            bins = [ self.richness_bins, self.redshift_bins], 
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
    if self.include_mwl_measurement_errors:
        mean_log10M_wl = mean_log10M_wl + np.random.normal( loc = 0.0 , scale = self.mwl_std )
        
    return np.vstack( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten()  

def stacked_counts_wonky_bins( self , richness, log10M_wl, redshift ):
    """
    Calculate the number of clusters in bins of cluster richness and redshift,
    and calculate the mean cluster weak-lensing mass in these bins.
    
    For KiDs where the richness bins change with redshift for the mass estimates 
    and they are defined completely different for the cluster counts

    Parameters:
    richness (array-like): Array of richness values.
    log10M_wl (array-like): Array of weak-lensing mass values (log10 scale).
    redshift (array-like): Array of redshift values.
    richness_bins (array-like): Bin edges for richness.
    redshift_bins (array-like): Bin edges for redshift.

    Returns:
    observed_cluster_abundance (1D array): Number of clusters in each bin.
    mean_log10M_wl (1D array): Mean log10 weak-lensing mass in each bin.
    """
    
    
    # counts....
    # Compute the 2D histogram for cluster counts
    observed_cluster_abundance, _, _ = np.histogram2d(
        richness, 
        redshift, 
        bins=[ self.richness_bins, self.redshift_bins ]
    )
    
    observed_cluster_abundance = observed_cluster_abundance.flatten()
    
    mean_log10M_wl = []
    # mass measurements...
    for idx_bin in np.arange( 0 , len( self.wl_redshift_bins[:-1] ) ):
        
        redshift_selection = ( redshift > self.wl_redshift_bins[idx_bin] ) & ( redshift < self.wl_redshift_bins[idx_bin+1] )
        
        in_observed_cluster_abundance, _ = np.histogram( richness[ redshift_selection ] , 
                                                            bins = self.wl_richness_bins[ idx_bin ] )
        
        sum_log10M_wl, _ = np.histogram( richness[ redshift_selection ] ,
                                            weights = ( 10**log10M_wl[ redshift_selection ] )**(1/3) ,
                                            bins = self.wl_richness_bins[idx_bin] )
        
        with np.errstate(divide='ignore', invalid='ignore'):
            in_mean_log10M_wl = np.where( in_observed_cluster_abundance > 0, 
                                            sum_log10M_wl / in_observed_cluster_abundance, 
                                            -1 )
            in_mean_log10M_wl = np.log10( in_mean_log10M_wl**3 )
        

        # add measurement and systematic uncertainties
        if self.include_mwl_measurement_errors:
            in_mean_log10M_wl = in_mean_log10M_wl + np.random.normal( loc = 0.0 , scale = self.mwl_std[idx_bin] )
            
        mean_log10M_wl.append( in_mean_log10M_wl )

    mean_log10M_wl = np.array( list( itertools.chain.from_iterable( mean_log10M_wl ) ) )
                
    return np.concatenate( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten()  

def old_stacked_counts( self , richness, log10M_wl, redshift ):
    """
    Calculate the number of clusters in bins of cluster richness and redshift,
    and calculate the mean cluster weak-lensing mass in these bins.

    Parameters:
    richness (array-like): Array of richness values.
    log10M_wl (array-like): Array of weak-lensing mass values (log10 scale).
    redshift (array-like): Array of redshift values.
    richness_bins (array-like): Bin edges for richness.
    redshift_bins (array-like): Bin edges for redshift.

    Returns:
    observed_cluster_abundance (2D array): Number of clusters in each bin.
    mean_log10M_wl (2D array): Mean log10 weak-lensing mass in each bin.
    """
    # Compute the 2D histogram for cluster counts
    observed_cluster_abundance, _, _ = np.histogram2d(
        richness, 
        redshift, 
        bins=[ self.richness_bins, self.redshift_bins]
    )


    # Compute the 2D histogram for the sum of log10M_wl
    sum_log10M_wl, _, _ = np.histogram2d(  richness, 
                                            redshift, 
                                            bins=[ self.richness_bins, self.redshift_bins], 
                                            weights=( 10**log10M_wl )
    )

    # Calculate mean log10M_wl in each bin (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_log10M_wl = np.where( observed_cluster_abundance > 0, 
                                    sum_log10M_wl / observed_cluster_abundance, 
                                    -1 )
        
        mean_log10M_wl = np.log10( mean_log10M_wl )
        

    return np.vstack( [ observed_cluster_abundance, mean_log10M_wl ] ).flatten()  