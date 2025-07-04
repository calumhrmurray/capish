import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/pbs/home/c/cmurray/cluster_likelihood/')
import modules.simulation as simulation
import pyccl as ccl
import modules.summary_statistics.des_summary_statistics as ss

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import sys
import pickle

if len(sys.argv) < 2:
    print("Usage: python run_sbi.py <config_path>")
    sys.exit(1)

config_path = sys.argv[1]

simulator = simulation.UniverseSimulator(
    ss.counts_and_mean_mass,
    config_path=config_path
)

# make sure that this is the correct size
prior = utils.BoxUniform( low = [ 0.05 , 0.5  , 1 , 0.5 ] ,
                          high = [ 1.0 , 1.5  , 2 , 1.0  ] )

posterior_estimator = infer( simulator.run_simulation ,
                       prior,
                       method = "SNPE",
                       num_simulations = 50 ,
                       num_workers = 20 )

if len(sys.argv) < 3:
    print("Usage: python run_sbi.py <config_path> <output_suffix>")
    sys.exit(1)

output_suffix = sys.argv[2]
output_path = f'/sps/euclid/Users/cmurray/clusters_likelihood/posterior_calculator_{output_suffix}.pkl'

with open(output_path, "wb") as handle:
    pickle.dump(posterior_estimator, handle)