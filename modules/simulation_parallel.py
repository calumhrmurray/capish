"""
Parallel simulation wrapper for SBI workflows

This module provides multiprocessing capabilities for running many
Capish simulations in parallel, which is essential for efficient
simulation-based inference (SBI).
"""

import numpy as np
import pickle
from multiprocessing import Pool
from tqdm.auto import tqdm
from pathlib import Path
import time
from modules.simulation import UniverseSimulator


class ParallelSimulator:
    """
    Wrapper around UniverseSimulator for parallel batch execution.

    This class handles multiprocessing, NaN filtering, checkpointing,
    and progress tracking for large simulation batches needed in SBI.
    """

    def __init__(self, config_path=None, default_config=None, variable_params_names=None):
        """
        Initialize the parallel simulator.

        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        default_config : ConfigParser, optional
            Configuration object (if config_path not provided)
        variable_params_names : list of str
            Names of parameters that will vary across simulations
        """
        self.config_path = config_path
        self.default_config = default_config
        self.variable_params_names = variable_params_names

        # Store for reference (actual simulators created in workers)
        self.simulator_template = UniverseSimulator(
            default_config_path=config_path,
            default_config=default_config,
            variable_params_names=variable_params_names
        )

    def _worker_init(self):
        """
        Initialize worker process with its own simulator instance.
        This is called once per worker process to avoid pickling issues.
        """
        global _worker_simulator
        _worker_simulator = UniverseSimulator(
            default_config_path=self.config_path,
            default_config=self.default_config,
            variable_params_names=self.variable_params_names
        )

    @staticmethod
    def _run_single_simulation(theta):
        """
        Worker function to run a single simulation.

        Parameters
        ----------
        theta : array-like
            Parameter vector for this simulation

        Returns
        -------
        tuple
            (theta, summary_stats, success_flag)
            If simulation fails (NaN/Inf), summary_stats will be None
        """
        try:
            # Use the worker's simulator instance
            summary_stats = _worker_simulator.run_simulation(theta)

            # Check for NaN or Inf
            if isinstance(summary_stats, tuple):
                # Handle tuple output (counts, masses)
                flat_stats = np.concatenate([np.array(s).flatten() for s in summary_stats])
            else:
                flat_stats = np.array(summary_stats).flatten()

            if np.any(~np.isfinite(flat_stats)):
                return (theta, None, False)

            return (theta, summary_stats, True)

        except Exception as e:
            # Log error but don't crash the worker
            print(f"Simulation failed for theta={theta}: {e}")
            return (theta, None, False)

    def run_batch_parallel(self, theta_batch, n_cores=20, checkpoint_path=None,
                          checkpoint_interval=1000, desc="Running simulations"):
        """
        Run a batch of simulations in parallel.

        Parameters
        ----------
        theta_batch : array-like, shape (n_sims, n_params)
            Batch of parameter vectors to simulate
        n_cores : int
            Number of CPU cores to use
        checkpoint_path : str or Path, optional
            If provided, save checkpoints to this file
        checkpoint_interval : int
            Save checkpoint every N simulations
        desc : str
            Description for progress bar

        Returns
        -------
        dict
            Dictionary containing:
            - 'theta': array of successful parameter vectors
            - 'x': array of successful summary statistics
            - 'failed_theta': array of failed parameter vectors
            - 'n_total': total simulations attempted
            - 'n_success': number of successful simulations
            - 'n_failed': number of failed simulations
            - 'success_rate': fraction of successful simulations
            - 'elapsed_time': total time in seconds
        """
        start_time = time.time()

        # Convert to list for pool.imap
        theta_list = [theta_batch[i] for i in range(len(theta_batch))]

        results = {
            'theta': [],
            'x': [],
            'failed_theta': [],
        }

        # Run simulations in parallel with progress bar
        with Pool(processes=n_cores, initializer=self._worker_init) as pool:
            for i, (theta, summary_stats, success) in enumerate(
                tqdm(pool.imap(self._run_single_simulation, theta_list),
                     total=len(theta_list),
                     desc=desc)
            ):
                if success:
                    results['theta'].append(theta)
                    results['x'].append(summary_stats)
                else:
                    results['failed_theta'].append(theta)

                # Checkpoint if requested
                if checkpoint_path and (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(results, checkpoint_path, i + 1)

        # Convert lists to arrays
        results['theta'] = np.array(results['theta'])
        results['failed_theta'] = np.array(results['failed_theta'])

        # Handle summary statistics (might be tuples or arrays)
        if results['x']:
            # Check if first result is a tuple
            if isinstance(results['x'][0], tuple):
                # Convert each element of tuple separately
                n_outputs = len(results['x'][0])
                x_arrays = []
                for i in range(n_outputs):
                    x_i = np.array([x[i] for x in results['x']])
                    x_arrays.append(x_i)
                results['x'] = tuple(x_arrays)
            else:
                results['x'] = np.array(results['x'])

        # Add statistics
        results['n_total'] = len(theta_batch)
        results['n_success'] = len(results['theta'])
        results['n_failed'] = len(results['failed_theta'])
        results['success_rate'] = results['n_success'] / results['n_total']
        results['elapsed_time'] = time.time() - start_time

        return results

    def _save_checkpoint(self, results, checkpoint_path, n_completed):
        """Save intermediate results to checkpoint file."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            'results': results,
            'n_completed': n_completed,
            'timestamp': time.time()
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"\nCheckpoint saved: {n_completed} simulations completed")

    @staticmethod
    def load_checkpoint(checkpoint_path):
        """
        Load results from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file

        Returns
        -------
        dict
            Checkpoint data including 'results' and 'n_completed'
        """
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
