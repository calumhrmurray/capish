import numpy as np

def richness_mass_relation( mu , z , parameter_set  ):
    """
    Returns whatever you want as a cluster catalogue.
    """
    log10Mmin = parameter_set['log10mmin']
    B = parameter_set['b']
    alpha_l = parameter_set['alpha_l']
    beta_l = parameter_set['beta_l']
    z_p = parameter_set['z_p']

    Mmin = 10**log10Mmin
    M1 = 10**( B ) * Mmin
    M = ( np.exp( mu ) * 1e14 )
    mean_l = ( ( M - Mmin ) / ( M1 -  Mmin ) )**alpha_l * ( ( 1 + z ) / ( 1 + z_p ) )**beta_l

    mean_l[ np.logical_or( mean_l < 0, np.isnan(mean_l) ) ] = 0

    return np.log( np.random.poisson( lam = mean_l ) + 1 )

def mass_observable_relation( mu, z, parameter_set ):

    sigma_l = parameter_set['sigma_l']
    r = parameter_set['r']
    sigma_mwl = parameter_set['sigma_mwl']  

    mean_l = richness_mass_relation( mu , z , parameter_set )
    mean_mu_mwl = mu

    cov = [ [ sigma_l**2 , r * sigma_l * sigma_mwl ], 
            [ r * sigma_l * sigma_mwl, sigma_mwl**2 ] ]

    total_noise = np.random.multivariate_normal([0, 0], cov=cov, size=len(mean_l))

    # Apply intrinsic noise to mean values
    ln_richness = mean_l + total_noise.T[0]
    mu_mwl = mean_mu_mwl + total_noise.T[1]

    return {
        'richness': np.exp(ln_richness),
        'log10_mwl': np.log10( np.exp( mu_mwl ) * 1e14 ),
        'redshift': z
    }

def sinh_mass_observable_relation(mu, z, parameter_set):
    sigma_l = parameter_set['sigma_l']
    sigma_mwl = parameter_set['sigma_mwl']

    def custom_sinh(mu):
        a = np.arcsinh(3)
        b = 14
        return np.sinh(a * (mu - b)) / 5

    mean_l = richness_mass_relation(mu, z, parameter_set)
    mean_mu_mwl = mu

    # Vectorized r calculation and clipping
    r = custom_sinh(mu)
    r = np.clip(r, -1, -0.8)

    # Vectorized Cholesky decomposition for covariance matrices
    cov00 = sigma_l ** 2
    cov11 = sigma_mwl ** 2
    cov01 = r * sigma_l * sigma_mwl

    # Build lower-triangular Cholesky factors for each sample
    L00 = np.sqrt(cov00)
    L10 = cov01 / L00
    L11 = np.sqrt(cov11 - L10 ** 2)

    # Generate standard normal samples
    n = len(mu)
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)

    # Apply Cholesky factors to get correlated noise
    noise0 = L00 * z1
    noise1 = L10 * z1 + L11 * z2

    ln_richness = mean_l + noise0
    mu_mwl = mean_mu_mwl + noise1

    return {
        'richness': np.exp(ln_richness),
        'log10_mwl': np.log10(np.exp(mu_mwl) * 1e14),
        'redshift': z
    }


MASS_OBSERVABLE_RELATIONS = {
    "default": mass_observable_relation,
    "sinh": sinh_mass_observable_relation,
}

class ClusterCatalogue:
     
    def __init__( self , settings ):
        """
        Initialize the ClusterCatalogue class with the given settings.
        """
        relation = settings["cluster_catalogue"]["mass_observable_relation"]
        self.mass_observable_relation = MASS_OBSERVABLE_RELATIONS[relation]

    def get_cluster_catalogue( self , halo_catalogue , parameter_set ):
        """
        Generate a cluster catalogue based on the halo catalogue and parameter set.
        """
        # Extract necessary parameters from the parameter set
        mu = halo_catalogue['mu']
        z = halo_catalogue['redshift']

        # Call the mass observable relation function
        return self.mass_observable_relation( mu, z, parameter_set )