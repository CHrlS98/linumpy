#!/usr/bin/env python3
"""
Apply corrections from linum_estimate_slices_transforms_gui.py to volume.
"""
import argparse
import zarr
import nibabel as nib
import numpy as np
from tqdm import tqdm

from linumpy.stitching.manual_registration import transform_slice, apply_scaling


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input',
                   help='Input file to correct (zarr or nifti).')
    p.add_argument('in_corrections',
                   help='File (.npz) containing the correction parameters.')
    p.add_argument('output',
                   help='Output zarr or nifti file.')
    p.add_argument('--rescale_off', action='store_true',
                   help='Flag to disable rescaling of intensities.')
    return p


def apply_transform(ty, tx, theta, coordinates):
    # Step 1. Rotate coordinates
    center_y = np.max(coordinates[:, :, 1]) / 2.0
    center_x = np.max(coordinates[:, :, 2]) / 2.0
    coordinates = coordinates - np.reshape([0, center_y, center_x], (1, 1, 3))
    rotated_y = np.atleast_2d(np.cos(theta)).T*coordinates[..., 1]\
        - np.atleast_2d(np.sin(theta)).T*coordinates[..., 2]
    rotated_x = np.atleast_2d(np.sin(theta)).T*coordinates[..., 1]\
        + np.atleast_2d(np.cos(theta)).T*coordinates[..., 2]
    coordinates[:, :, 1] = rotated_y + center_y
    coordinates[:, :, 2] = rotated_x + center_x

    # Step 2. Translate coordinates
    coordinates[:, :, 1] += np.atleast_2d(ty).T
    coordinates[:, :, 2] += np.atleast_2d(tx).T

    return coordinates


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    checkpoint = np.load(args.in_corrections)
    custom_ranges = checkpoint['custom_ranges']
    transforms = checkpoint['transforms']

    if '.zarr' in args.input:
        in_data = zarr.open(args.input, mode='r')
    else:
        in_nifti = nib.load(args.input)
        in_data = in_nifti.get_fdata()

    if '.zarr' in args.output:
        out_data = zarr.open(args.output, mode='w',
                             shape=in_data.shape,
                             dtype=in_data.dtype)
    else:
        out_data = np.zeros_like(in_data)

    imin, imax = np.min(in_data), np.max(in_data)

    # process slices one at a time
    for z in tqdm(range(in_data.shape[0])):
        # estimate_slices_transforms_gui rescales intensities between (0, 1).
        data = in_data[z]
        transform_z = transforms[z]
        ranges_z = custom_ranges[z]
        ty = transform_z[0]
        tx = transform_z[1]
        theta = transform_z[2]
        vmin = ranges_z[0]
        vmax = ranges_z[1]
        transformed_image = transform_slice(data, ty, tx, theta)
        if not args.rescale_off:
            transformed_image = (transformed_image - imin) / (imax - imin)
            transformed_image = apply_scaling(transformed_image, vmin, vmax)
        out_data[z] = transformed_image

    if '.nii' in args.output:
        nib.save(nib.Nifti1Image(out_data, in_nifti.affine),
                 args.output)


if __name__ == '__main__':
    main()
