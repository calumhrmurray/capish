import numpy as np
import pyccl as ccl
import itertools
import sys, os
import copy
import configparser
from halo.halo_catalogue import HaloCatalogue
from cluster.cluster_catalogue import ClusterCatalogue
from summary_statistics.summary_statistics import SummaryStatistics
import utils as utils

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

        config = copy.deepcopy(self.default_config)
        for i, name in enumerate(self.variable_params_names):
            config['parameters'][str(name)] = str(variable_params_values[i])
        return config

    def run_simulation_halo_catalogue(self, variable_params_values):
        """
        Run the simulation using the variable parameters provided  - only halo catalogue.
        """

        config = self.new_config_files(variable_params_values)
        halo_data = self.halo_catalogue_class.get_halo_catalogue( config )
        log10m_halo = halo_data['log10mass']
        z_true = halo_data['redshift']

        return log10m_halo, z_true

    def run_simulation_cluster_catalogue(self, variable_params_values):
        """
        Run the simulation using the variable parameters provided.
        """

        config = self.new_config_files(variable_params_values)
        halo_data = self.halo_catalogue_class.get_halo_catalogue( config )
        log10m_halo = halo_data['log10mass']
        z_true = halo_data['redshift']
        richness, log10mWL, z_obs = self.cluster_catalogue_class.get_cluster_catalogue( log10m_halo, z_true , config )

        return richness, log10mWL, z_obs

    def run_simulation_from_halo_properties(self, log10m_halo, z_true, variable_params_values):
        """
        Run the simulation using the variable parameters provided.
        """
        np.random.seed(12345)
        config = self.new_config_files(variable_params_values)
        richness, log10mWL, z_obs = self.cluster_catalogue_class.get_cluster_catalogue( log10m_halo, z_true , config )
        summary_statistic = self.summary_statistics_class.get_summary_statistics( richness, log10mWL, z_obs, config )

        return summary_statistic

    def run_simulation(self, variable_params_values):
        """
        Run the simulation using the variable parameters provided.
        """

        config = self.new_config_files(variable_params_values)
        richness, log10mWL, z_obs = self.run_simulation_cluster_catalogue(variable_params_values)
        summary_statistic = self.summary_statistics_class.get_summary_statistics( richness, log10mWL, z_obs, config )

        return summary_statistic
