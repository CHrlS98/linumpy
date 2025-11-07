#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import nibabel as nib
from skimage.filters import frangi
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input image.')
    p.add_argument('out_image',
                   help='Output filtered image.')
    p.add_argument('--scale_range', nargs=2, type=float, default=[1, 10],
                   help='Range of sigma used. [%(default)s]')
    p.add_argument('--n_scales', type=int, default=10,
                   help='Number of scales. Must be greater than 1. [%(default)s]')
    p.add_argument('--alpha', type=float, default=0.5,
                   help='Correction constant that adjusts sensitivity to\n'
                        'deviation from a plate-like structure. [%(default)s]')
    p.add_argument('--beta', type=float, default=0.5,
                   help='Correction constant that adjusts sensitivity to\n'
                        'deviation from a blob-like structure. [%(default)s]')
    p.add_argument('--gamma', type=float,
                   help='Correction constant that adjusts the sensitivity to areas\n'
                        'of high variance/texture/structure. By default, half of the\n'
                        'maximum Hessian norm.')
    p.add_argument('--black_ridges', action='store_true', default=False,
                   help='When True, the filter detects black ridges.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_im = nib.load(args.in_image)
    data = in_im.get_fdata()

    sigmas = np.linspace(args.scale_range[0], args.scale_range[1], args.n_scales)
    out = frangi(data, sigmas, alpha=args.alpha, beta=args.beta,
                 gamma=args.gamma, black_ridges=args.black_ridges)
    nib.save(nib.Nifti1Image(out.astype(np.float32), in_im.affine), args.out_image)


if __name__ == '__main__':
    main()
