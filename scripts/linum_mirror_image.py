#!/usr/bin/env python3
"""
Generate "mirror" brain image by reversing data
about the sagittal (YZ) plane.
"""
import argparse
import nibabel as nib
import numpy as np

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_nifti',
                   help='Input nifti image in RAS orientation.')
    p.add_argument('out_nifti',
                   help='Output file, mirrored about the YZ plane.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    im = nib.load(args.in_nifti)
    data = im.get_fdata()
    data = data[::-1, ...]
    nib.save(nib.Nifti1Image(data, im.affine), args.out_nifti)


if __name__ == '__main__':
    main()
