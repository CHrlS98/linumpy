#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split nifti image into two hemispheres at a given index.
"""
import argparse
import nibabel as nib
from linumpy.utils.io import assert_output_exists, add_overwrite_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image', help='Input nifti image.')
    p.add_argument('index', type=int, help='Index at which to split the image.')
    p.add_argument('out_until', help='Output nifti image from 0 to index.')
    p.add_argument('out_from', help='Output nifti image from index to end of volume.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_exists(args.out_until, parser, args)
    assert_output_exists(args.out_from, parser, args)

    in_image = nib.load(args.in_image)
    image_until = in_image.slicer[:args.index]
    nib.save(image_until, args.out_until)
    del image_until  # free memory before loading the second half of the image

    image_from = in_image.slicer[args.index:]
    nib.save(image_from, args.out_from)


if __name__ == '__main__':
    main()
