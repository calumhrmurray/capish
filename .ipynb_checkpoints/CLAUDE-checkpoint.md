# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cosmology research codebase focused on cluster likelihood analysis and simulation-based inference (SBI). The project implements theoretical models for galaxy cluster abundance, mass-observable relations, and likelihood calculations for cosmological parameter estimation.

## Architecture

The codebase is organized into several key directories:

### `/modules/`
Core Python modules containing the scientific computing logic:

- **`halo/`** - Halo-related computations using CCL (Core Cosmology Library):
  - `halo_catalogue.py` - Main `HaloCatalogue` class with configurable mass functions (Despali16, Tinker10) and bias models
  - `_halo_abundance.py` - Halo abundance calculations and super-sample covariance

- **`cluster/`** - Modular cluster catalog system:
  - `cluster_catalogue.py` - Main `ClusterCatalogue` class orchestrating the full pipeline
  - `_halo_observable_relation.py` - `HaloToObservables` class implementing mass-observable relations with power-law scaling
  - `_completeness.py` - Detection completeness modeling
  - `_purity.py` - Sample purity and contamination modeling  
  - `_selection.py` - Survey selection functions

- **`summary_statistics/`** - Summary statistics for cosmological inference:
  - `summary_statistics.py` - Main `SummaryStatistics` class with binned counts and mean mass calculations
  - `des_summary_statistics.py` - DES-specific implementations

- **`simulation.py`** - Main `UniverseSimulator` class coordinating the full simulation pipeline
- **`simulation_with_multiprocessing.py`** - Multiprocessing version for large simulation suites

### `/notebooks/`
Research notebooks organized by researcher:
- `calum/` - SBI experiments and analysis notebooks
- `constantin/` - Validation, forecasting, and consistency check notebooks

### `/config/`
Configuration files in INI format:
- `capish.ini` - Main parameter configuration with unified parameter structure
- `capish_sinh.ini` - Alternative configuration with sinh-based correlation models

### `/old/`
Legacy notebooks and experimental code

### `/pinocchio/`
Simulation data and analysis tools

## Key Dependencies

The project relies heavily on:
- **PyCC** (Core Cosmology Library) - For cosmological calculations
- **NumPy/SciPy** - Numerical computations
- **SBI libraries** - For simulation-based inference workflows

## Configuration System

The project uses INI-style configuration files with the following structure:
- `[parameters]` - Unified section containing both cosmological and mass-observable relation parameters
- `[halo_catalogue]` - Mass function, bias model, super-sample covariance, and survey configuration
- `[cluster_catalogue]` - Completeness, purity, and selection settings
- `[cluster_catalogue.mass_observable_relation]` - Mass-observable relation type specification  
- `[cluster_catalogue.photometric_redshift]` - Photometric redshift error parameters
- `[summary_statistics]` - Data vector configuration with richness, redshift, and mass binning

## Workflow Patterns

The simulation pipeline follows this modular structure:

1. **Halo Catalog Generation** (`HaloCatalogue`):
   - Uses CCL with configurable mass functions (Despali16, Tinker10) and bias models
   - Supports super-sample covariance (SSC) with optional precomputed covariance matrices
   - Outputs true halo masses and redshifts: `{log10m_true, z_true}`

2. **Observable Generation** (`ClusterCatalogue`):
   - Applies completeness cuts based on true mass and redshift
   - Generates observables (richness, weak lensing mass, photometric redshift) via `HaloToObservables`
   - Adds contamination via purity modeling
   - Applies survey selection functions
   - Outputs: `{richness, log10mWL, z_obs}`

3. **Summary Statistics** (`SummaryStatistics`):
   - Computes binned number counts and mean masses in richness-redshift-mass bins
   - Supports various binning schemes and lensing weight calculations
   - Outputs data vectors for cosmological inference

4. **Full Pipeline** (`UniverseSimulator`):
   - Orchestrates the complete simulation from cosmological parameters to summary statistics
   - Supports both single simulations and multiprocessing for large parameter sweeps

## Development Notes

- All scientific modules should follow the existing pattern of using CCL for cosmological calculations
- Configuration changes should be made through INI files rather than hardcoding parameters
- New mass functions or bias models should be added to the respective maps in `halo_catalogue.py`
- The modular cluster pipeline allows independent testing and modification of completeness, purity, and selection functions
- Mass-observable relations use power-law scaling with configurable pivot points and scatter parameters
- Summary statistics support both traditional binned counts and mean mass estimates for enhanced cosmological constraints

## Git Commit Guidelines

- NEVER add Claude as a co-author in commit messages
- Do not include "Co-Authored-By: Claude <noreply@anthropic.com>" in commits
- Keep commit messages concise and descriptive of actual changes made