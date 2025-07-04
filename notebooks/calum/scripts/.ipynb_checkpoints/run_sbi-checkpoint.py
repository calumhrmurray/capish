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

simulator = simulation.UniverseSimulator( ss.counts_and_mean_mass ,  config_path = '/pbs/home/c/cmurray/cluster_likelihood/config/capish.ini' )

# make sure that this is the correct size
prior = utils.BoxUniform( low = [ 0.05 , 0.5  , 1 , 0.5 ] ,
                          high = [ 1.0 , 1.5  , 2 , 1.0  ] )

des_posterior = infer( simulator.run_simulation ,
                       prior,
                       method = "SNPE",
                       num_simulations = 500 ,
                       num_workers = 20 )