#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linum_split_nifti.py
This script is designed to split a NIfTI image into multiple slices
along a specified axis. It takes an input NIfTI file and outputs a series
of nifti files, each containing a single slice of the original image.
"""
import os
import shutil
import numpy as np
import nibabel as nib
import argparse


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_nifti",
                     help="Path to the input NIfTI file.")
    p.add_argument("output_dir",
                        help="Directory to save the output NIfTI slices.")
    p.add_argument('--axis', choices=['sagittal', 'coronal', 'axial'],
                    default='sagittal',
                    help='Axis along which to split the image [%(default)s].')
    p.add_argument('--output_prefix', default='slice_',
                    help='Prefix for the output slice files [%(default)s].')
    p.add_argument('-f', dest='overwrite', action='store_true',
                    help='Overwrite output directory if it exists.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the input NIfTI file
    in_image = nib.load(args.input_nifti)

    # Check if the output directory exists, if not create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif not args.overwrite:
        parser.error(f"Output directory {args.output_dir} already exists. Use -f to overwrite.")
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Determine the axis along which to split the image
    if args.axis == 'sagittal':
        axis = 0
    elif args.axis == 'coronal':
        axis = 1
    elif args.axis == 'axial':
        axis = 2

    # A slice is considered valid if 5% of its area is non-zero
    min_nb_voxels = np.prod(in_image.shape) / in_image.shape[axis] * 0.05
    print(f"Minimum number of non-zero voxels for a slice to be valid: {min_nb_voxels}")

    _slice = [slice(s) for s in in_image.shape]

    # Save each slice as a separate NIFTI file
    for i in range(in_image.shape[axis]):
        # Only save the slice if it has non-zero data
        _slice[axis] = slice(i, i+1)
        slice_img = in_image.slicer[tuple(_slice)]
        slice_data = slice_img.get_fdata()
        if np.count_nonzero(slice_data > 1E-4) > min_nb_voxels:
            output_filename = f"{args.output_dir}/{args.output_prefix}{i}.nii.gz"
            nib.save(nib.Nifti1Image(np.squeeze(slice_data), slice_img.affine),
                     output_filename)
            print(f"Saved slice {i} to {output_filename}")


if __name__ == "__main__":
    main()
