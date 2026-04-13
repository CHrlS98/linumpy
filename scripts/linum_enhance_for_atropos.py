#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
"""
import argparse
import nibabel as nib
import numpy as np
from skimage.exposure import equalize_adapthist


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('in_oct', help='Input oct image.')
    p.add_argument('in_mask', help='Input brain mask.')
    p.add_argument('out_image', help='Output nifti image.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_oct = nib.load(args.in_oct)
    in_oct_data = in_oct.get_fdata().astype(np.float32)

    in_mask = nib.load(args.in_mask)
    in_mask_data = in_mask.get_fdata().astype(np.float32)

    # rescale both OCT and AMBA between 0 and 1
    in_oct_data = (in_oct_data - in_oct_data.min()) / (in_oct_data.max() - in_oct_data.min())

    # CLAHE equalization of OCT image
    out_data = equalize_adapthist(in_oct_data, (25, 25, 25), clip_limit=0.01)

    out_data = out_data * in_mask_data

    nib.save(nib.Nifti1Image(out_data.astype(np.float32), in_oct.affine), args.out_image)


if __name__ == '__main__':
    main()
