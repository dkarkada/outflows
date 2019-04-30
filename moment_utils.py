'''
    Module for analyzing moments of wavelength cubes.

    Author: Aaron T. Lee, Spring/Summer 2018
            Dhruva Karkada, Spring 2019

    Free to share, use, blah blah.
    I take no responsibility if any errors that exist below screw up your
    analysis.
    Don't treat code like a black box.
'''

import datetime
import numpy as np
from astropy.io import fits


# Global set of constants (cgs)
LIGHT_SPEED = 3.0e10
BOLTZMANN_CONST = 1.3806e-16
LINE_CM = .2600757633465  # CO line peak wavelength


# convert intensity cube to temperature cube
def create_temp_cube(intensity_cube, camera_wavelengths):
    # remember that RADMC outputs B_nu despite using wavelengths
    def intensity_to_temp(intensity_nu, freq):
        return intensity_nu*pow(LIGHT_SPEED/freq, 2)/(2.0*BOLTZMANN_CONST)
    temp_cube = []
    for wave_ind in range(len(camera_wavelengths)):
        freq = LIGHT_SPEED / camera_wavelengths[wave_ind]
        wave_image = intensity_to_temp(intensity_cube[wave_ind, :, :], freq)
        temp_cube.append(wave_image)
    return np.array(temp_cube)


# Writes a moment map to a FITS file
def save_moment_map(moment_map, outfile=None):
    if outfile is None:
        outfile = "{}-moment.fits".format(datetime.datetime.now().isoformat())
    print("Saving moment map in FITS format to {}.".format(outfile))
    hdu = fits.PrimaryHDU(moment_map)
    hdu.writeto(outfile, overwrite=True)


# Reads in moment map from a FITS file
def read_moment_map(infile="moment.fits"):
    print("Reading {} as FITS.".format(infile))
    with fits.open(infile) as hdu_list:
        hdu = hdu_list[0]
        return hdu.data
    assert False, "Could not open {}.".format(infile)


# Creates a 1D spectrum for the given pixel(s)
def create_spectrum(data_cube, pixel=None, moment_map=None):
    # if no coordinates provided, generate spectrum at max-intensity pixel
    if pixel is None:
        assert moment_map is not None, "Must provide moment map."
        print("Searching for maximum intensity pixel...")
        x, y = np.unravel_index(moment_map.argmax(), moment_map.shape)
        print("Max intensity at pixel x={},y={}".format(x, y))
        pixel = (x, y)

    # generate spectra
    _, dimx, dimy = data_cube.shape
    x, y = pixel
    if x >= dimx or x < 0 or y >= dimy or y < 0:
        raise ValueError("Index out of bound for this file: x={}, y={}, \
                            dim = ({},{})".format(x, y, dimx, dimy))
    return data_cube[:, x, y]


# creates doppler velocity (km/s) moment map
def create_moment_map(data_cube, camera_wavelengths, moment=0, mean=None):
    print("Creating {}-moment map.".format(moment))

    # compute mean if necessary
    if mean is None and moment >= 2:
        print("Calculating mean map.")
        moment0 = create_moment_map(data_cube, camera_wavelengths, moment=0)
        moment1 = create_moment_map(data_cube, camera_wavelengths, moment=1)
        mean = normalize_map(moment1, moment0)

    # convert wavelengths to doppler velocities
    velocities = LIGHT_SPEED * (camera_wavelengths - LINE_CM) / LINE_CM
    # convert to km/s
    velocities /= 1.0e5
    # make velocities broadcastable
    bcastable_vel = velocities[:, None, None]
    # get central moment for high-order moments. -= operator doesn't work
    if moment >= 2:
        bcastable_vel = bcastable_vel - mean
    # compute moment. Note 0**0=1 in python
    moment_cube = data_cube * (bcastable_vel**moment)
    # integration needs positive dv, so flip x-axes if necessary
    if velocities[0] > velocities[-1]:
        velocities = np.flip(velocities)
        moment_cube = np.flip(moment_cube, axis=0)
    moment_map = np.trapz(moment_cube, x=velocities, axis=0)

    print("Done creating moment map. Enjoy.")
    return moment_map


def normalize_map(moment_map, norm_map):
    print("Normalizing.")
    assert moment_map.shape == norm_map.shape, "norm_map sized incorrectly"
    norm_map += np.min(norm_map[np.nonzero(norm_map)]) / 1e6
    return moment_map / norm_map
