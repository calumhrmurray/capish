import numpy as np
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
logm_edges = np.linspace(14.3, 15.5, 5)
redshift_edges = np.linspace(0.2, 1, 6)
richness_edges = np.exp(np.linspace(np.log(20), np.log(200), 6))
Z_bin = binning(redshift_edges)
LogMass_bin = binning(logm_edges)
Richness_bin = binning(richness_edges)