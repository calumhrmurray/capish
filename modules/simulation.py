import numpy as np
import pyccl as ccl
import itertools
import sys
import os 
import configparser
from modules.halo.halo_catalogue import HaloCatalogue
from modules.cluster.cluster_catalogue import ClusterCatalogue

class UniverseSimulator:
    
    def __init__( self , summary_statistic , config_path = None ):
        """
        Initialize the UniverseSimulator class.
        """

        if config_path:
            config = configparser.ConfigParser()
            config.read(config_path)

            # here we setup the parameters which will be used for the simulation
            # also set the fixed parameters which will not be varied
            self.variable_params = list(config['variable_parameters'].keys())
            self.fixed_params = {k: float(v) for k, v in config['fixed_parameters'].items()}

            # set the halo catalogue settings
            # self.halo_catalogue_settings = {k: float(v) for k, v in config['halo_catalogue'].items()}
            self.halo_catalogue_class = HaloCatalogue( config )

            # set the cluster catalogue settings
            #self.cluster_catalogue_settings = {k: float(v) for k, v in config['cluster_catalogue'].items()}
            #self.cluster_catalogue_class = ClusterCatalogue( self.cluster_catalogue_settings )

            # set the summary statistic function
            # this should function on a cluster catalogue object
            #self.summary_statistic = summary_statistic

        else:
            print("No config file provided, you must provide a config.")


    def run_simulation( self , param_values ):
        """
        Run the simulation using the variable parameters provided.
        """
        
        halo_catalogue = self.halo_catalogue_class.get_halo_catalogue( param_values )

        cluster_catalogue = self.cluster_catalogue_class.get_cluster_catalogue( halo_catalogue , param_values )

        summary_statistic = self.get_summary_statistic( cluster_catalogue )

        return summary_statistic

    def _get_parameter_set(self, param_values):
        """
        Create the full parameter set by combining fixed and variable parameters.
        """
        # Start with default parameter values
        parameter_set = self.available_params.copy()

        # Assign the passed variable parameters to their corresponding keys
        for i, param in enumerate( self.variable_params ):
            parameter_set[param] = float(param_values[i])  # Ensure float type

        # Overwrite with any fixed parameters
        parameter_set.update( self.fixed_params)

        # Return a dictionary of the parameters
        return parameter_set

    def run_simulation(self, param_values):
        """
        Run the simulation using the variable parameters provided.
        """
        # Get the full parameter set (both variable and fixed parameters)
        parameter_set = self._get_parameter_set( param_values )

        # Run the core simulation
        richness, log10M_wl, z_clusters, _  = self._run_simulation( parameter_set )

        # Return result in a format compatible with SBI
        if self.for_simulate_for_sbi:
            return self.summary_statistic(richness, log10M_wl, z_clusters)
        #     return torch.tensor( self.summary_statistic(richness, log10M_wl, z_clusters))
        else:
            return self.summary_statistic(richness, log10M_wl, z_clusters)

    