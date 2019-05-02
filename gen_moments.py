'''
    Script to generate moments of wavelength cubes.

    Author: Dhruva Karkada, Spring 2019

    Free to share, use, blah blah.
    I take no responsibility for any errors that exist here.
'''

import numpy as np
from astropy.io import fits
import moment_utils as mutil

LIGHT_SPEED = 3.0e10
BOLTZMANN_CONST = 1.3806e-16
LINE_CM = .2600757633465  # CO line peak wavelength


def observer_outflow_mask(data_cube, camera_wavelengths):
    # convert wavelengths to doppler velocities
    velocities = LIGHT_SPEED * (camera_wavelengths - LINE_CM) / LINE_CM
    # convert to km/s
    velocities /= 1.0e5
    mask = abs(velocities) > 1
    # make mask broadcastable
    bcastable_mask = mask[:, None, None]
    return bcastable_mask

# Full cube moment maps
head, wave, tempcube = mutil.get_temp_cube("full.fits")
moment0_map = mutil.create_moment_map(tempcube, wave, moment=0)
mutil.save_map(moment0_map, "full-mom0.fits")
moment1_map = mutil.create_moment_map(tempcube, wave, moment=1)
mean = mutil.normalize_map(moment1_map, moment0_map)
moment2_map = mutil.create_moment_map(tempcube, wave, moment=2, mean=mean)
vrms_map = np.sqrt(mutil.normalize_map(moment2_map, moment0_map))
mutil.save_map(vrms_map, "full-vrms.fits")

# Tracer cells moment maps
head, wave, tempcube = mutil.get_temp_cube("tracer.fits")
moment0_map = mutil.create_moment_map(tempcube, wave, moment=0)
mutil.save_map(moment0_map, "tracer-mom0.fits")
moment2_map = mutil.create_moment_map(tempcube, wave, moment=2, mean=mean)
vrms_map = np.sqrt(mutil.normalize_map(moment2_map, moment0_map))
mutil.save_map(vrms_map, "tracer-vrms.fits")

# Non-tracer cells moment maps
head, wave, tempcube = mutil.get_temp_cube("nontracer.fits")
moment0_map = mutil.create_moment_map(tempcube, wave, moment=0)
mutil.save_map(moment0_map, "nontracer-mom0.fits")
moment2_map = mutil.create_moment_map(tempcube, wave, moment=2, mean=mean)
vrms_map = np.sqrt(mutil.normalize_map(moment2_map, moment0_map))
mutil.save_map(vrms_map, "nontracer-vrms.fits")

# Observer's outflow moment maps
head, wave, tempcube = mutil.get_temp_cube("full.fits")
mask = observer_outflow_mask(tempcube, wave)
tempcube_outflow_masked = tempcube * mask
obs_outflow0 = mutil.create_moment_map(tempcube_outflow_masked, wave, moment=0)
mutil.save_map(obs_outflow0, "obs-outflow-mom0.fits")
obs_outflow2 = mutil.create_moment_map(tempcube_outflow_masked, wave,
                                       moment=2, mean=mean)
vrms_map = np.sqrt(mutil.normalize_map(obs_outflow2, obs_outflow0))
mutil.save_map(vrms_map, "obs-outflow-vrms.fits")

# Observer's non-outflow moment maps
mask = np.logical_not(mask)
tempcube_nonoutflow_masked = tempcube * mask
obs_nonoutflow0 = mutil.create_moment_map(tempcube_nonoutflow_masked,
                                          wave, moment=0)
mutil.save_map(obs_nonoutflow0, "obs-nonoutflow-mom0.fits")
obs_nonoutflow2 = mutil.create_moment_map(tempcube_nonoutflow_masked,
                                          wave, moment=2, mean=mean)
vrms_map = np.sqrt(mutil.normalize_map(obs_nonoutflow2, obs_nonoutflow0))
mutil.save_map(vrms_map, "obs-nonoutflow-vrms.fits")
