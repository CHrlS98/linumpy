#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image')
    p.add_argument('out_image')
    p.add_argument('xmin', type=int)
    p.add_argument('xmax', type=int)
    p.add_argument('ymin', type=int)
    p.add_argument('ymax', type=int)
    p.add_argument('zmin', type=int)
    p.add_argument('zmax', type=int)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = nib.load(args.in_image)
    data = in_image.get_fdata()
    data_cropped = data[args.xmin:args.xmax,
                        args.ymin:args.ymax,
                        args.zmin:args.zmax]

    # same affine as input
    nib.save(nib.Nifti1Image(data_cropped, in_image.affine),
             args.out_image)


if __name__ == '__main__':
    main()
