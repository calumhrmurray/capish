import numpy as np
import pyccl as ccl
from scipy import stats

def ngal_arcmin2_to_Mpc2(ngal_arcmin2, Da_Mpc):
    """
    Convert galaxy number density from arcmin⁻² to Mpc⁻² given an angular diameter distance.

    Parameters
    ----------
    ngal_arcmin2 : float or array_like
        Galaxy number density in units of galaxies per square arcminute.
    Da_Mpc : float or array_like
        Angular diameter distance to the lens in megaparsecs (Mpc).

    Returns
    -------
    ngal_Mpc2 : float or np.ndarray
        Galaxy number density in units of galaxies per square Mpc.
    """
    return ngal_arcmin2 * (180*60/(np.pi*Da_Mpc))**2

def mean_sigma_crit(z_lens, cosmo, zmax=5.0):
    """
    Compute the mean critical surface density Σ_crit for a lens at redshift `z_lens`,
    averaged over a source galaxy redshift distribution (Chang et al. 2013).

    Parameters
    ----------
    z_lens : float
        Redshift of the lens.
    cosmo : ccl.Cosmology
        Fiducial cosmology object from pyccl.
    zmax : float, optional
        Maximum source redshift to consider. Default is 5.0.

    Returns
    -------
    mean_Sigma_crit : float
        Mean critical surface density [M_sun / pc² or consistent units with Σ_crit function].
    """
    z_s = np.linspace(z_lens + 0.2, zmax, 500)
    nz = nz_chang2013(z_s)
    sigma_crit_2 = sigma_crit(cosmo, z_lens, z_s) ** (-2)

    numerator = np.trapz(nz * sigma_crit_2, z_s)
    denominator = np.trapz(nz, z_s)

    return (numerator / denominator) ** (-0.5)

def nz_chang2013(z, alpha=2.0, beta=1.5, z0=0.5):
    return z ** alpha * np.exp(-(z / z0)**beta)

def sigma_epsilon(z, sigma_e=0.3):
    return sigma_e * np.ones(len(z))  # constant or e.g., sigma_e * (1 + z)

def sigma_crit(cosmo, z_l, z_s):
    return ccl.sigma_critical(cosmo, a_lens=1/(1+z_l), a_source=1/(1+z_s))

def lensing_weights(cosmo, z_l_array, z_s_max=5.0, n_zs=500, sigma_e_const=0.3):

    """
    Compute lensing efficiency weights W(z_l) for an array of lens redshifts.

    The weight W(z_l) is defined as the average inverse variance of the critical surface density 
    Σ_crit for source galaxies behind a lens at redshift z_l, weighted by the source redshift 
    distribution and intrinsic shape noise.

    Parameters
    ----------
    cosmo : ccl.Cosmology
        Fiducial cosmology object from pyccl.
    z_l_array : array_like
        Array of lens redshifts for which to compute the weights.
    z_s_max : float, optional
        Maximum source redshift to consider. Default is 5.0.
    n_zs : int, optional
        Number of points in the source redshift grid. Default is 500.
    sigma_e_const : float, optional
        RMS intrinsic ellipticity of source galaxies. Default is 0.3.

    Returns
    -------
    weights : np.ndarray
        Array of lensing weights W(z_l), same shape as `z_l_array`.
    """
    
    z_l_array = np.atleast_1d(z_l_array)
    weights = np.zeros_like(z_l_array)

    # Source redshift grid
    z_s = np.linspace(0.01, z_s_max, n_zs)
    n_z = nz_chang2013(z_s)
    sigma_e = sigma_epsilon(z_s, sigma_e_const)

    for i, z_l in enumerate(z_l_array):
        mask = z_s > z_l
        z_s_masked = z_s[mask]
        n_z_masked = n_z[mask]
        sigma_e_masked = sigma_e[mask]

        if len(z_s_masked) == 0:
            weights[i] = 0.0
            continue

        sigma_crit_vals = sigma_crit(cosmo, z_l, z_s_masked)
        sigma_crit_sq_inv = 1.0 / sigma_crit_vals**2
        integrand = n_z_masked * sigma_crit_sq_inv / sigma_e_masked**2

        numerator = np.trapz(integrand, z_s_masked)
        denominator = np.trapz(n_z_masked, z_s_masked)
        weights[i] = numerator / denominator if denominator > 0 else 0.0
    return weights

def excess_surface_density_fct(R, m, concentration, z, cosmo_ccl,
                               mdef='mean', delta_mdef=200):
    """
    Compute the excess surface density ΔΣ(R) for a single halo using a given concentration-mass profile.

    Parameters
    ----------
    R : array_like
        Array of projected radial distances [Mpc/h] at which to compute ΔΣ.
    m : float
        Halo mass [M_sun/h] at redshift `z`.
    concentration : float
        Halo concentration parameter.
    z : float
        Redshift of the halo.
    cosmo_ccl : ccl.Cosmology
        Cosmology object from pyccl.
    mdef : str, optional
        Mass definition type, e.g., 'mean' or 'critical'. Default is 'mean'.
    delta_mdef : float, optional
        Overdensity parameter for the halo mass definition. Default is 200.

    Returns
    -------
    excess_surface_density : np.ndarray
        Array of ΔΣ values at the radial points specified in `R`.
    """
    import clmm
    from clmm import Cosmology
    clmm_cosmology = Cosmology()
    clmm_cosmology.be_cosmo = cosmo_ccl
    excess_surface_density = clmm.compute_excess_surface_density(R, m, 
                                                        concentration, z, 
                                                        clmm_cosmology, delta_mdef=delta_mdef,
                                                        halo_profile_model='nfw',
                                                        massdef=mdef)
    return excess_surface_density

def model_error_log10m_one_cluster(log10m_grid, z_grid,
                                                cosmo, Rmin=1,Rmax=3,
                                                ngal_arcmin2=25,shape_noise=0.25,
                                                delta=200,mass_def='critical',
                                                cM='Duffy08'):

    """
    Compute the weak-lensing 1σ error on log10(M) for a single cluster as a function of halo mass 
    and redshift, using a Fisher matrix approach with ΔΣ (excess surface density) measurements.
    
    Parameters
    ----------
    log10m_grid : array_like
        Array of log10 halo masses [M_sun/h] to evaluate.
    z_grid : array_like
        Array of redshifts at which to evaluate the mass error.
    cosmo : ccl.Cosmology
        Fiducial cosmology object from pyccl.
    Rmin : float, optional
        Minimum projected radius for ΔΣ computation [Mpc/h]. Default is 1.
    Rmax : float, optional
        Maximum projected radius for ΔΣ computation [Mpc/h]. Default is 3.
    ngal_arcmin2 : float, optional
        Galaxy number density per square arcminute for shape noise computation. Default is 25.
    shape_noise : float, optional
        RMS of intrinsic galaxy ellipticity. Default is 0.25.
    delta : float, optional
        Overdensity parameter for halo mass definition (e.g., 200 for 200c). Default is 200.
    mass_def : str, optional
        Mass definition type: 'critical' or 'mean'. Default is 'critical'.
    cM : str, optional
        Concentration-mass relation model. Options include:
        'Diemer15', 'Duffy08', 'Prada12', 'Bhattacharya13', 'Klypin11'. Default is 'Duffy08'.

    Returns
    -------
    sigma_log10M : np.ndarray
        2D array of 1σ errors on log10(M), shape (len(log10m_grid), len(z_grid)).
    """
    
    # --- Concentration model setup ---
    deff = ccl.halos.massdef.MassDef(delta, mass_def)
    if cM == 'Diemer15':
        conc_model = ccl.halos.concentration.ConcentrationDiemer15(mass_def=deff)
    elif cM == 'Duffy08':
        conc_model = ccl.halos.concentration.ConcentrationDuffy08(mass_def=deff)
    elif cM == 'Prada12':
        conc_model = ccl.halos.concentration.ConcentrationPrada12(mass_def=deff)
    elif cM == 'Bhattacharya13':
        conc_model = ccl.halos.concentration.ConcentrationBhattacharya13(mass_def=deff)
    elif cM == 'Klypin11':
        conc_model = ccl.halos.concentration.ConcentrationKlypin11(mass_def=deff)
    else:
        raise ValueError(f"Unknown concentration model: {cM}")

    # --- Radial grid ---
    n_radial_bins = 500
    R = np.geomspace(Rmin, Rmax, n_radial_bins)

    # --- Allocate arrays ---
    excess_surface_density_m_mdm_R = np.zeros((len(log10m_grid), len(z_grid), n_radial_bins))
    excess_surface_density_m_pdm_R = np.zeros((len(log10m_grid), len(z_grid), n_radial_bins))
    concentration = np.zeros((len(log10m_grid), len(z_grid)))
    mean_Sigma_crit = np.zeros(len(z_grid))

    # --- Source redshift distribution ---
    z_s = np.linspace(0, 5, 1000)
    integral_full_nz = np.trapz(nz_chang2013(z_s), z_s)
    full_nz_normalized = nz_chang2013(z_s) / integral_full_nz

    prob_background_zs = np.tile(full_nz_normalized, (len(z_grid), 1))
    for j, z in enumerate(z_grid):
        prob_background_zs[j, z_s < (z + 0.2)] = 0

    prob_background = np.trapz(prob_background_zs, z_s, axis=1)
    Da = ccl.angular_diameter_distance(cosmo, 1 / (1 + z_grid))

    ngal_arcmin2_background = ngal_arcmin2 * prob_background
    ngal_Mpc2_background = ngal_arcmin2_to_Mpc2(ngal_arcmin2_background, Da)

    # --- Concentration & Sigma_crit ---
    for j, z in enumerate(z_grid):
        mean_Sigma_crit[j] = mean_sigma_crit(z, cosmo, zmax=5.0)
        concentration[:, j] = conc_model._concentration(cosmo, 10 ** log10m_grid, 1.0 / (1.0 + z))

    # --- Compute ΔΣ for slightly perturbed masses ---
    Dm = np.zeros((len(log10m_grid), len(z_grid)))
    for i, log10m in enumerate(log10m_grid):
        for j, z in enumerate(z_grid):
            conc = concentration[i, j]
            Dm[i, j] = 0.02 * (10 ** log10m)
            mass_plus = 10 ** log10m + Dm[i, j]
            mass_minus = 10 ** log10m - Dm[i, j]

            excess_surface_density_m_mdm_R[i, j, :] = excess_surface_density_fct(
                R, mass_plus, conc, z, cosmo, mdef='critical', delta_mdef=200
            )
            excess_surface_density_m_pdm_R[i, j, :] = excess_surface_density_fct(
                R, mass_minus, conc, z, cosmo, mdef='critical', delta_mdef=200
            )

    # --- Derivative wrt mass ---
    Dm_expanded = Dm[:, :, np.newaxis]
    derivative_DS_dm = (excess_surface_density_m_mdm_R - excess_surface_density_m_pdm_R) / (2 * Dm_expanded)

    # --- Shape noise error term ---
    error_DS_per_R_2_SN = (mean_Sigma_crit ** 2 * shape_noise ** 2) / ngal_Mpc2_background  # [Σ_crit^2 * σ_γ^2 / n_gal]
    
    # Expand this along radius and mass
    error_DS_per_R_2_SN_expanded = error_DS_per_R_2_SN[np.newaxis, :, np.newaxis]

    # --- Fisher information integrand ---
    Fm_to_integrate = derivative_DS_dm ** 2 * (error_DS_per_R_2_SN_expanded ** -1) * (2 * np.pi * R)

    # --- Integrate over radius ---
    F_mm = np.trapz(Fm_to_integrate, R, axis=2)

    # --- Return 1σ error on log10(M) ---
    Log10m_grid = np.tile(log10m_grid[:, np.newaxis], (1, len(z_grid)))
    sigma_log10M = (F_mm ** -0.5) / (np.log(10) * (10 ** Log10m_grid))

    return sigma_log10M