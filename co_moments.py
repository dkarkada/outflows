import numpy as np
from astropy.io import fits
import moment_utils as mutil

LIGHT_SPEED = 3.0e10
BOLTZMANN_CONST = 1.3806e-16
LINE_CM = .2600757633465  # CO line peak wavelength


def get_temp_cube(fits_file):
    header = {}
    data_cube = camera_wavelengths = None
    with fits.open(fits_file) as hdu_list:
        # load image data
        image_hdu = hdu_list[0]
        for key, val in image_hdu.header.items():
            header[key] = val
        data_cube = image_hdu.data
        # load wavelength data (in centimeters)
        assert len(hdu_list) >= 0, "FITS must contain both image \
                                    and wavelength data!"
        camera_wavelengths = hdu_list[1].data["wavelengths"]

        assert data_cube is not None, "Data not found."
        assert camera_wavelengths is not None, "Wavelengths not found."
    temp_cube = mutil.create_temp_cube(data_cube, camera_wavelengths)

    return header, camera_wavelengths, temp_cube


def observer_outflow_mask(data_cube, camera_wavelengths):
    # convert wavelengths to doppler velocities
    velocities = LIGHT_SPEED * (camera_wavelengths - LINE_CM) / LINE_CM
    # convert to km/s
    velocities /= 1.0e5
    mask = abs(velocities) > 1
    # make mask broadcastable
    bcastable_mask = mask[:, None, None]
    return bcastable_mask

# Tracer cells moment maps
head, wave, tempcube = get_temp_cube("tracer.fits")
moment0_map = mutil.create_moment_map(tempcube_full, wave_full, moment=0)
mutil.save_moment_map(moment0_map, "tracer-mom0.fits")
moment2_map = mutil.create_moment_map(tempcube_full, wave_full, moment=2)
mutil.save_moment_map(moment2_map, "tracer-mom2.fits")

# Non-tracer cells moment maps
head, wave, tempcube = get_temp_cube("nontracer.fits")
moment0_map = mutil.create_moment_map(tempcube_full, wave_full, moment=0)
mutil.save_moment_map(moment0_map, "nontracer-mom0.fits")
moment2_map = mutil.create_moment_map(tempcube_full, wave_full, moment=2)
mutil.save_moment_map(moment2_map, "nontracer-mom2.fits")

# Full cube moment maps
head, wave, tempcube = get_temp_cube("full.fits")
moment0_map = mutil.create_moment_map(tempcube_full, wave_full, moment=0)
mutil.save_moment_map(moment0_map, "full-mom0.fits")
moment2_map = mutil.create_moment_map(tempcube_full, wave_full, moment=2)
mutil.save_moment_map(moment2_map, "full-mom2.fits")

# Observer's outflow moment maps
mask = observer_outflow_mask(tempcube, wave)
tempcube_outflow_masked = tempcube * mask
obs_outflow0 = mutil.create_moment_map(tempcube_outflow_masked, wave, moment=0)
mutil.save_moment_map(obs_outflow0, "obs-outflow-mom0.fits")
obs_outflow2 = mutil.create_moment_map(tempcube_outflow_masked, wave, moment=2)
mutil.save_moment_map(obs_outflow2, "obs-outflow-mom2.fits")

# Observer's non-outflow moment maps
mask = np.logical_not(mask)
tempcube_nonoutflow_masked = tempcube * mask
obs_nonoutflow0 = mutil.create_moment_map(tempcube_nonoutflow_masked,
                                          wave, moment=0)
mutil.save_moment_map(obs_nonoutflow, "obs-nonoutflow-mom0.fits")
obs_nonoutflow2 = mutil.create_moment_map(tempcube_nonoutflow_masked,
                                          wave, moment=2)
mutil.save_moment_map(obs_nonoutflow, "obs-nonoutflow-mom2.fits")
