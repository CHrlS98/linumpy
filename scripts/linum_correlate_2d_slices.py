#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the correlation coefficient between two 2D NIfTI image and print
the result.
"""

import argparse
import numpy as np
import nibabel as nib

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_image_1",
                   help="Path to the first input NIfTI file.")
    p.add_argument("input_image_2",
                   help="Path to the second input NIfTI file.")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the input NIfTI files
    img1 = nib.load(args.input_image_1)
    img2 = nib.load(args.input_image_2)

    # Check if the images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")

    # Convert images to numpy arrays
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Compute the correlation coefficient
    correlation_matrix = np.corrcoef(data1.flatten(), data2.flatten())
    correlation_coefficient = correlation_matrix[0, 1]

    print(f"{correlation_coefficient}")


if __name__ == "__main__":
    main()
