#!/usr/bin/env python3
"""
Crop nifti volume between given indices.
"""
import argparse
import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image',
                   help='Input NIFTI image.')
    p.add_argument('out_image',
                   help='Output cropped image.')
    p.add_argument('--xmin', type=int,
                   help='Lower bound along first axis.')
    p.add_argument('--xmax', type=int,
                   help='Upper bound along first axis.')
    p.add_argument('--ymin', type=int,
                   help='Lower bound along second axis.')
    p.add_argument('--ymax', type=int,
                   help='Upper bound along second axis.')
    p.add_argument('--zmin', type=int,
                   help='Lower bound along last axis.')
    p.add_argument('--zmax', type=int,
                   help='Upper bound along last axis.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = nib.load(args.in_image)
    data = in_image.get_fdata()

    xmin = 0 if args.xmin is None else args.xmin
    xmax = data.shape[0] if args.xmax is None else args.xmax
    ymin = 0 if args.ymin is None else args.ymin
    ymax = data.shape[1] if args.ymax is None else args.ymax
    zmin = 0 if args.zmin is None else args.zmin
    zmax = data.shape[2] if args.zmax is None else args.zmax

    data_cropped = data[xmin:xmax, ymin:ymax, zmin:zmax]

    # same affine as input
    nib.save(nib.Nifti1Image(data_cropped, in_image.affine),
             args.out_image)


if __name__ == '__main__':
    main()
