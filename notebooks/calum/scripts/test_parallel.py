"""
Quick test script to verify parallel simulation works

This runs a small batch (10 simulations) to test the infrastructure
before running the full 10,000 simulation job.

Usage:
    python test_parallel.py
"""

import sys
import os
import time

# IMPORTANT: Set thread limits BEFORE importing numpy/scipy/pyccl
# This prevents each worker from spawning too many threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.abspath('../../../'))

import numpy as np
import configparser
from modules.simulation_parallel import ParallelSimulator
from modules.simulation import UniverseSimulator

# Test configuration
N_TEST_SIMS = 10
N_CORES = 4

print("="*70)
print("PARALLEL SIMULATION TEST")
print("="*70)
print(f"Test simulations: {N_TEST_SIMS}")
print(f"Cores: {N_CORES}")
print("="*70)
print()

# Load configuration (flagship-specific)
config = configparser.ConfigParser()
config.read('../../../config/capish_flagship.ini')

# Parameters to vary
variable_params_names = ['Omega_m', 'sigma8', 'alpha_lambda', 'beta_lambda']

# Test 1: Sequential simulation
print("Test 1: Running sequential simulations...")
simulator_seq = UniverseSimulator(default_config=config, variable_params_names=variable_params_names)

# Sample test parameters
np.random.seed(42)
theta_test = np.random.uniform(
    low=[0.25, 0.75, -9.5, 0.7],
    high=[0.35, 0.85, -9.0, 0.8],
    size=(N_TEST_SIMS, 4)
)

start_seq = time.time()
results_seq = []
for theta in theta_test:
    result = simulator_seq.run_simulation(theta)
    results_seq.append(result)
time_seq = time.time() - start_seq

print(f"Sequential time: {time_seq:.2f} seconds ({time_seq/N_TEST_SIMS:.2f} sec/sim)")
print()

# Test 2: Parallel simulation
print("Test 2: Running parallel simulations...")
simulator_par = ParallelSimulator(default_config=config, variable_params_names=variable_params_names)

start_par = time.time()
results_par = simulator_par.run_batch_parallel(
    theta_batch=theta_test,
    n_cores=N_CORES,
    desc="Test simulations"
)
time_par = time.time() - start_par

print()
print(f"Parallel time: {time_par:.2f} seconds ({time_par/N_TEST_SIMS:.2f} sec/sim)")
print(f"Speedup: {time_seq/time_par:.2f}x")
print(f"Efficiency: {(time_seq/time_par)/N_CORES*100:.1f}%")
print()

# Verify results
print("="*70)
print("VERIFICATION")
print("="*70)
print(f"Sequential successful: {N_TEST_SIMS}/{N_TEST_SIMS}")
print(f"Parallel successful: {results_par['n_success']}/{results_par['n_total']}")
print(f"Parallel success rate: {results_par['success_rate']:.1%}")
print()

# Check that results match (approximately)
if isinstance(results_seq[0], tuple):
    # Tuple output (counts, masses)
    for i in range(min(3, N_TEST_SIMS)):
        if i < len(results_par['x'][0]):
            seq_counts = np.array(results_seq[i][0]).flatten()
            par_counts = results_par['x'][0][i].flatten()

            match = np.allclose(seq_counts, par_counts, rtol=0.1)
            print(f"Sim {i}: Match = {match}")
            if not match:
                print(f"  Sequential: {seq_counts[:5]}")
                print(f"  Parallel:   {par_counts[:5]}")
else:
    # Single array output
    for i in range(min(3, N_TEST_SIMS)):
        if i < len(results_par['x']):
            seq_stat = np.array(results_seq[i]).flatten()
            par_stat = results_par['x'][i].flatten()

            match = np.allclose(seq_stat, par_stat, rtol=0.1)
            print(f"Sim {i}: Match = {match}")
            if not match:
                print(f"  Sequential: {seq_stat[:5]}")
                print(f"  Parallel:   {par_stat[:5]}")

print()
print("="*70)
print("TEST COMPLETE")
print("="*70)
print()
print("If speedup > 1 and results match, the parallel infrastructure is working!")
print(f"Expected speedup with 20 cores: ~{20 * 0.7:.0f}-{20 * 0.9:.0f}x (70-90% efficiency)")
print()
