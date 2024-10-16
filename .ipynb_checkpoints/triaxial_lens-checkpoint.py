import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
from astropy import constants
import sys
sys.path.insert( 0 , '/pbs/home/c/cmurray/selection_function/lensing/lensing_amplitudes/')

def memoize( func ):
	cache = dict()
	
	def memoized_func(*args):
		if args in cache:
			return cache[args]
		result = func(*args)
		cache[args] = result
		return result
	
	return memoized_func

# surface mass density {dimensionless}
def sigma_0( x ):

	# this is my way of dealing with the discontinuity at x=1...
	# if the change is smaller than this it does not help 
	if x < 1:
		return ( 2 / ( x**2 -1 ) ) * \
			   ( 1 - 2 * np.arctanh( np.sqrt( ( 1- x ) / ( 1 + x ) ) ) / (  1 - x**2 )**0.5 )
	#if x == 1:
	#    return 2/3
	#else:
	x = x + 7e-5 
	return ( 2 / ( x**2 -1 ) ) * \
			   ( 1 - 2 * np.arctan( np.sqrt( ( x - 1 ) / ( x + 1 ) ) ) / (  x**2  - 1)**0.5 )

sigma_0 = np.vectorize( sigma_0 )

def eta_0( x ):
	dx = 1e-6
	xh = x + dx
	xl = x - dx
	sh = np.log( sigma_0( xh ) )
	sl = np.log( sigma_0( xl ) )
	return x * ( sh - sl ) / ( 2 * dx )

def I_1_integrand( x ):
	return np.power( x , 3 ) * sigma_0( x ) * eta_0( x )

def I_1( x ):
	i = integrate.quad( I_1_integrand , 0 , x )[0]
	return i * 3 / x ** 4

def I_2_integrand( x ):
	return sigma_0( x ) * eta_0( x ) / x

def I_2( x ):
	i = integrate.quad( I_2_integrand , x , 5e2 )[0]
	return i

I_2 = np.vectorize( I_2 )

# does the heavy lifting for quad_shear_t
# this means we can memomize just for x
# should keep stuff fast
def __quad_shear_t__( x ):
	a = sigma_0( x ) * eta_0( x )
	return ( a -  I_1( x ) - I_2( x ) ) 

__quad_shear_t__ = np.vectorize( __quad_shear_t__ )

def __alt_qst__( x ):
	return np.interp( x , xx , qst )

def quad_shear_t( x , psi , e ):
	return ( e / 2 ) * __alt_qst__( x ) * np.cos( 2 * psi )

quad_shear_t = np.vectorize( quad_shear_t )

def __quad_shear_x__( x ):
	return ( -  I_1( x ) + I_2( x ) )

__quad_shear_x__ = np.vectorize( __quad_shear_x__ )

def __alt_qsx__( x ):
	return np.interp( x , xx , qsx )

def quad_shear_x( x , psi , e ):
	return ( e / 2 ) * __alt_qsx__( x ) * np.sin( 2 * psi )

def av_sigma_0_integrand( x ):
	return x * sigma_0( x )

def av_sigma_0( x ):
	i = integrate.quad( av_sigma_0_integrand , 0 , x )[0]
	return 2 * i / x ** 2

__av_sigma_0__ = np.vectorize( av_sigma_0 )


def __mono_shear_t__( x ):
	return av_sigma_0( x ) - sigma_0( x )

__mono_shear_t__ = np.vectorize( __mono_shear_t__ )

def __alt_mono_shear_t__( x ):
	return np.interp( x , xx , mst )

def mono_shear_t( x ):
	return __alt_mono_shear_t__( x )

# monopole + quadrupole
def shear_t( x , psi , e ):
	return mono_shear_t( x ) + quad_shear_t( x , psi , e )


# only the quadrupole is non zero
def shear_x( x , psi , e ):
	return quad_shear_x( x , psi , e )


# shear = np.vectorize( shear )

direc = ''

xx = np.logspace( -5 , 5  , num = 200 )
qst = np.load( direc + 'qst.npy')
qsx = np.load( direc + 'qsx.npy')
mst = np.load( direc + 'mst.npy')

# average sigma for deflection angle
as0 = __av_sigma_0__( xx )
#qst = __quad_shear_t__( xx )
#qsx = __quad_shear_x__( xx )
#mst = __mono_shear_t__( xx )

def av_sigma( x ):
	return np.interp( x , xx , as0 )


class triaxial_lens:
	
	def __init__( s , M_lens , c , z_lens , e , H0 = 70 , Om0 = 0.3 ):
		# lens mass [M_sun] 
		s.M_lens = M_lens
		# lens concentration
		s.c = c
		# lens redshift
		s.z_lens = z_lens
		# lens ellipticity
		s.e = e
		# set up the cosmology
		s.cosmo = FlatLambdaCDM( H0 = H0 , Om0 = Om0 )
		# critical density of the universe [M_sun / Mpc^3]
		s.rho_c = s.cosmo.critical_density( s.z_lens ).to('M_sun/Mpc^3').value
		# mean matter density of the universe [M_sun / Mpc^3]
		s.rho_m = s.rho_c * s.cosmo.Om( s.z_lens )
		# halo definition
		s.delta = 200
		# 
		s.delta_c = s.char_overdensity( c )
		# [Mpc]
		s.r200 = s.r200_calc()
		# scale radius [Mpc]
		s.rs = s.rs_calc()
		# scale radius radians
		s.theta_s = s.rs / s.cosmo.angular_diameter_distance( s.z_lens ).value
	
	# characteristic overdensity, dimensionless
	def char_overdensity( s , c ):
		return ( s.delta / 3. ) * ( c**3 / ( np.log( 1 + c ) - c / ( 1 + c ) ) )
	
	# calculate r 200
	def r200_calc( s ):
		r2003 = ( 3.0 * s.M_lens ) / ( 4.0 * np.pi * s.delta * s.rho_m )
		return r2003 ** ( 1. / 3. )
	
	# scale radius [Mpc]
	def rs_calc( s ):
		return s.r200 / s.c
	
	# this is the prefactor which gives the correct amplitude
	# to the signal calculated above
	def A( s ):
		return s.rs * s.rho_m * s.delta_c
								   
	# excess surface density
	# theta in radians
	# psi in radians
	# e ellipticity
	# units [M_sun/pc^2]
	def esd_t( s , theta , psi ):
		x = theta / s.theta_s
		# factor of 1e12 changes units from M_sun/Mpc^2 -> M_sun/pc^2
		return s.A() * shear_t( x , psi , s.e ) / 1e12


	def esd_x( s , theta , psi ):
		x = theta / s.theta_s
		# factor of 1e12 changes units from M_sun/Mpc^2 -> M_sun/pc^2
		return s.A() * shear_x( x , psi , s.e ) / 1e12

	def gamma_t( s , theta , psi , z_gal ):
		return s.esd_t( theta , psi ) / s.sigma_cr( s.z_lens , z_gal )

	# units M_sun/pc^2
	def sigma_cr( s , z_lens , z_source ):
		G = constants.G
		c = constants.c
		D_s = s.cosmo.angular_diameter_distance( z_source )
		D_d = s.cosmo.angular_diameter_distance( z_lens )
		D_ds = s.cosmo.angular_diameter_distance_z1z2( z_lens , z_source )
		return ( ( c**2 / (4*np.pi*G) ) * D_s / ( D_d * D_ds ) ).to('M_sun / pc^2').value

	# monopole convergence
	def kappa_0( s , theta , z_gal ):
		x = theta / s.theta_s
		return s.A() * sigma_0( x ) / 1e12 / s.sigma_cr( s.z_lens , z_gal )

	def sigma( s , theta  ):
		x = theta / s.theta_s
		return s.A() * sigma_0( x ) / 1e12 

	# the deflection angle in radians
    # is this correct? I would expect 1/theta
	def alpha( s, theta , z_gal ):
		x = theta / s.theta_s
		return s.A() * theta * av_sigma( x ) / s.sigma_cr( s.z_lens , z_gal ) / 1e12

	# dimensionless esd
	def template( s , theta ):
		x = theta / s.theta_s
		return shear_t( x , 0 , s.e )

	 

		

