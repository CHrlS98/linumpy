#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a 2D image to a 3D image sharing the spacial attributes of a reference image.
"""
import nibabel as nib
import numpy as np
import argparse


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_nifti",
                   help="Path to the input NIfTI file.")
    p.add_argument("reference_nifti",
                   help="Path to the reference NIfTI file.")
    p.add_argument("output_nifti",
                   help="Path to the output NIfTI file.")
    p.add_argument('--axis', type=int, default=0,
                   help='Axis along which to add a length-1 dimension [%(default)s].')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the input NIfTI file
    in_image = nib.load(args.input_nifti)
    ref_image = nib.load(args.reference_nifti)

    # Add a length-1 dimension along the specified axis
    new_shape = list(in_image.shape)
    new_shape.insert(args.axis, 1)
    out_data = np.reshape(in_image.get_fdata(), new_shape)

    # Create a new NIfTI image with the modified data
    out_image = nib.Nifti1Image(out_data, ref_image.affine, ref_image.header)

    # Save the output NIfTI file
    nib.save(out_image, args.output_nifti)


if __name__ == "__main__":
    main()
