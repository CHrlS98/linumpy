#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter, convolve
from linumpy.utils.coordinates import AXIS_NAME_TO_INDEX, slice_along_axis

def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image')
    p.add_argument('out_image')
    p.add_argument('--sigma', type=float, default=3.0)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = nib.load(args.in_image)
    data = in_image.get_fdata()
    data_filtered = gaussian_filter(data, sigma=args.sigma)
    gradient_filter = np.array([1.0, 0.0, -1.0])
    dx = convolve(data_filtered, gradient_filter.reshape((-1, 1, 1)))
    dy = convolve(data_filtered, gradient_filter.reshape((1, -1, 1)))
    dz = convolve(data_filtered, gradient_filter.reshape((1, 1, -1)))
    gradient_mag = dx**2 + dy**2 + dz**2

    nib.save(nib.Nifti1Image(gradient_mag, in_image.affine), args.out_image)


if __name__ == '__main__':
    main()
