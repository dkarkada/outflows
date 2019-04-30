import argparse
import numpy as np
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument('-if', '--infile', help='radmc image file \
                    (default=image.out)', default='image.out', type=str)
parser.add_argument('-of', '--outfile', help='fits file (default=image.fits)',
                    default='image.fits', type=str)


def main():
    # Parse the command line
    args = parser.parse_args()

    print("Converting {} to {}...".format(args.infile, args.outfile))
    # Reads the image.out file from line emission computation
    data_cube = None
    image_hdu = fits.PrimaryHDU()
    header = image_hdu.header
    wave_table_hdu = None
    with open(args.infile, 'r') as f:
        line = f.readline()
        header["IFORMAT"] = np.int64(line.strip())

        line = f.readline()
        header["NX"], header["NY"] = dimx, dimy = [
            np.int64(x) for x in line.strip().split()]

        line = f.readline()
        header["NWAVS"] = num_wavelengths = np.int64(line.strip())

        line = f.readline()
        header["PIXSZ_X"], header["PIXSZ_Y"] = [
            np.float64(x) for x in line.strip().split()]

        # Let's make sure we are in 'observer at infinity' mode
        assert header["IFORMAT"] == 1, "Not in 'observer at infinity' mode"

        # The next set of lines will be the camera wavelengths.
        camera_wavelengths = []
        for _ in range(num_wavelengths):
            line = f.readline()
            wavelength = np.float64(line.strip())
            camera_wavelengths.append(wavelength)
        # convert to cgs
        camera_wavelengths = 1.0e-4*np.array(camera_wavelengths)
        # make fits table
        wave_table_hdu = fits.BinTableHDU.from_columns([
                            fits.Column(name="wavelengths",
                                        format="E", array=camera_wavelengths)])

        # Now read all the actual data
        data_cube = np.empty((num_wavelengths, dimx, dimy))
        line_num = 0
        for line in f:
            line = line.strip()
            if line == "":
                continue
            wave_ind, x_ind, y_ind = np.unravel_index(
                line_num, (num_wavelengths, dimx, dimy))
            data_cube[wave_ind, x_ind, y_ind] = np.float64(line)
            line_num += 1

    image_hdu.data = data_cube
    hdu_list = fits.HDUList([image_hdu, wave_table_hdu])
    hdu_list.writeto(args.outfile, overwrite=True)
    print("All done.")


if __name__ == "__main__":
    main()
