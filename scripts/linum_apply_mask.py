#!/usr/bin/env python3
"""
Replace data inside mask by interpolating data
from previous and next slices.
"""
import argparse
from scipy.ndimage import binary_dilation
from scipy.interpolate import griddata, CubicSpline
import nibabel as nib
import numpy as np
from linumpy.utils.coordinates import AXIS_NAME_TO_INDEX
from tqdm import tqdm
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_nifti')
    p.add_argument('in_mask')
    p.add_argument('out_nifti')
    p.add_argument('--fill_value', type=float, default=0.0)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = nib.load(args.in_nifti)
    data = in_image.get_fdata()
    mask = nib.load(args.in_mask).get_fdata().astype(bool)

    data[mask] = args.fill_value

    nib.save(nib.Nifti1Image(data, in_image.affine),
             args.out_nifti)


if __name__ == '__main__':
    main()
