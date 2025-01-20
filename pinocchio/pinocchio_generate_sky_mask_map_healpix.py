from astropy.io import fits
from astropy.table import Table, hstack
import sys, glob
import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord

#define the function to be used
def cat_to_hpx(lon, lat, nside, radec=True):
    
    """
    Converts a catalogue to a HEALPix map of number density i.e. 
    Number of stars per square degrees of sky.

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. 
        If radec=True, assume input is in the icrs system,
        Otherwise assume input is galactic latitude and longitude.


    nside : int
        HEALPix nside of the target map, defines the number of pixels.

    radec : bool
        Switch between Ra/Dec and l/b (galactic) as input coordinate system.

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates

    """

    npix = hp.nside2npix(nside)

    if radec:
        eq = SkyCoord(lon, lat, unit='deg')
        l, b = eq.galactic.l.value, eq.galactic.b.value
    else:
        l, b = lon, lat
        
    # convert to theta, phi -> galactic longitude and colatitude in sphererical system
    theta = np.radians(90. - b)
    phi = np.radians(l)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    indx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[indx] = counts
    hpx_map = np.where(hpx_map != 0, 1, 0)

    return hpx_map

where_pinocchio_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/'
file=glob.glob(where_pinocchio_cat+'*')
nside = 256
mask_map = np.zeros(hp.nside2npix(nside), dtype=int)
for i in range(500):
    print('pinocchio simulation number = ' + str(i))
    dat = pd.read_csv(file[i] ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
    ra, dec, redshift, Mvir_true = dat['ra'], dat['dec'], dat['z'], dat['M']/0.6777
    mask_map += cat_to_hpx(ra, dec, nside, radec=True)
    mask_map = np.where(mask_map != 0, 1, 0)
    hp.write_map("../data/pinocchio_mask_map_sky_coverage/pinocchio_mask_map_sky_coverage.fits", mask_map, overwrite=True)