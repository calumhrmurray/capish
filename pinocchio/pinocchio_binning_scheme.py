import numpy as np
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
redshift_edges = np.linspace(0.2, 1, 6)
logm_edges = np.linspace(13.5, 16, 16)
richness_edges = np.exp(np.linspace(np.log(20), np.log(200), 6))
Z_bin = binning(redshift_edges)
LogMass_bin = binning(logm_edges)
Richness_bin = binning(richness_edges)