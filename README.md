# Capish

Capish is a modular Python pipeline designed to simulate and analyze cosmological cluster data. It builds mock catalogues of dark matter halos and galaxy clusters based on cosmological parameters, models selection and measurement processes, and computes summary statistics relevant for cosmological inference.

---

## Overview

Capish is organized into three main stages:

---

### I. Halo Catalogue Generation

**File:** `halo_catalogue.py`  
**Class:** `HaloCatalogue`

Generates a mock catalogue of dark matter halos using input cosmological parameters. This includes computing:

- Halo bias  
- Halo mass function  
- Volume  
- Super-sample covariance (SSC) matrix

**Output:** {m_halo, z_true}


**Dependencies:**

- `_halo_abundance.py`: Implements halo abundance models

---

### II. Cluster Catalogue Generation

**File:** `cluster_catalogue.py`  
**Class:** `ClusterCatalogue`

Simulates observable properties of galaxy clusters from the true halo catalogue. Incorporates models for:

- Weak lensing mass (`m_WL`)
- Richness (`λ_obs`)
- Photometric redshift (`z_phot`)

**Output:** {m_WL, λ_obs, z_phot}_sel

**Dependencies:**

- `completeness.py`: `P(detect | m_true, z_true)`
- `halo_observable_relation.py`: `P(m_WL, λ_obs, z_phot | m_halo, z_true)`
- `purity.py`: `P(fake | λ_obs, z_phot)`
- `selection.py`: Cluster sample selection logic

---

### III. Summary Statistics

**File:** `summary_statistics.py`  
**Class:** `SummaryStatistics`

Computes summary statistics of the simulated cluster catalogue:

- Binned number counts: `N_ijk`
- Mass estimates: `M_ij`

**Output:** {N_ijk} or {N_ij, M_ij}

### IV. Basic usage:

- `simulator = simulation.UniverseSimulator(
    default_config_path = '../../config/capish.ini',
    variable_params_names = ['Omega_m', 'sigma_8'])`

- `new_config = simulator.new_config_files([0.25, 0.8])`

- `log10m_true, z_true = simulator.halo_catalogue_class.get_halo_catalogue(new_config)`

- `richness, log10mWL, z_obs = simulator.cluster_catalogue_class.get_cluster_catalogue(
    log10m_true, z_true, new_config)`

- `count, mean_mass = simulator.summary_statistics_class.get_summary_statistics(
    richness, log10mWL, z_obs, new_config)`







