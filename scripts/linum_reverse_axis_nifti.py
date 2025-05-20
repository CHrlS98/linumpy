#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverse a nifti volume along a given axis.
"""
import argparse
import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_nifti',
                     help='Input nifti image to reverse.')
    p.add_argument('out_nifti',
                     help='Reversed nifti image.')
    p.add_argument('--axis', choices=['sagittal', 'coronal', 'axial'],
                   default='sagittal',
                   help='Axis along which to reverse the image.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = nib.load(args.in_nifti)
    image_data = in_image.get_fdata()

    # Reverse the image data along the specified axis
    axis_map = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    axis = axis_map[args.axis]
    reversed_image_data = np.flip(image_data, axis=axis)

    # Save the reversed image
    out_image = nib.Nifti1Image(reversed_image_data, in_image.affine, in_image.header)
    nib.save(out_image, args.out_nifti)


if __name__ == "__main__":
    main()
