import numpy as np
import pyccl as ccl
import itertools
import sys
import os 
import copy
import configparser
from modules.halo.halo_catalogue import HaloCatalogue
from modules.cluster.cluster_catalogue import ClusterCatalogue
from modules.summary_statistics.summary_statistics import SummaryStatistics

class UniverseSimulator:
    
    def __init__(self, default_config_path = None , default_config = None, variable_params_names = None):
        """
        Initialize the UniverseSimulator class.
        """

        if (default_config_path != None) + (default_config != None):
            if (default_config_path != None):
                default_config = configparser.ConfigParser()
                default_config.read(default_config_path)

            elif default_config != None:
                default_config = default_config

            self.params_names = list(default_config['parameters'].keys())
            self.params_values = {k: float(v) for k, v in default_config['parameters'].items()}
            self.variable_params_names = variable_params_names
            self.default_config = default_config
            
            self.halo_catalogue_class = HaloCatalogue( default_config )
            self.cluster_catalogue_class = ClusterCatalogue( default_config )
            self.summary_statistics_class = SummaryStatistics( default_config )

        else: print("No config file provided, you must provide a config.")

    def new_config_files(self, variable_params_values):

        config_new = copy.deepcopy(self.default_config)
        for i, name in enumerate(self.variable_params_names):
            config_new['parameters'][str(name)] = str(variable_params_values[i])
        return config_new


    def run_simulation(self, variable_params_values):
        """
        Run the simulation using the variable parameters provided.
        """

        config_new = self.new_config_files(variable_params_values)
        log10m_true, z_true = self.halo_catalogue_class.get_halo_catalogue( config_new )
        richness, log10mWL, z_obs = self.cluster_catalogue_class.get_cluster_catalogue( log10m_true, z_true , config_new )
        summary_statistic = self.summary_statistics_class.get_summary_statistics( richness, log10mWL, z_obs, config_new )

        return summary_statistic

#     #def _get_parameter_set(self, param_values):
#      #   """
#      #   Create the full parameter set by combining fixed and variable parameters.
#      #   """
#      #   # Start with default parameter values
#       #  parameter_set = self.available_params.copy()
# #
#         # Assign the passed variable parameters to their corresponding keys
#       #  for i, param in enumerate( self.variable_params ):
#       #      parameter_set[param] = float(param_values[i])  # Ensure float type

#         # Overwrite with any fixed parameters
#       #  parameter_set.update( self.fixed_params)
# #
#         # Return a dictionary of the parameters
#       #  return parameter_set

#   #  def run_simulation(self, param_values):
#    #     """
#    #@     Run the simulation using the variable parameters provided.
#    #     """
#    #     # Get the full parameter set (both variable and fixed parameters)
#    #     parameter_set = self._get_parameter_set( param_values )

#         # Run the core simulation
#         richness, log10M_wl, z_clusters, _  = self._run_simulation( parameter_set )

#         # Return result in a format compatible with SBI
#         if self.for_simulate_for_sbi:
#             return self.summary_statistic(richness, log10M_wl, z_clusters)
#         #     return torch.tensor( self.summary_statistic(richness, log10M_wl, z_clusters))
#         else:
#             return self.summary_statistic(richness, log10M_wl, z_clusters)

    