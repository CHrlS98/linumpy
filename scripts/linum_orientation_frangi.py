#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Fiber orientations analysis using Frangi filters as in [1].
"""
import argparse
import nibabel as nib
import numpy as np

from linumpy.feature.foa3d import frangi_filter as frangi_foa3d
from linumpy.feature.frangi import frangi_filter as frangi_skimage


EPILOG="""
[1] Sorelli et al, 2023, "Fiber enhancement and 3D orientation analysis in label-free
    two-photon fluorescence microscopy", Scientific Reports (2023) 13:4160
"""

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image', help='Input nifti image.')
    p.add_argument('out_prefix', help='Output images prefix.')
    p.add_argument('--alpha', default=0.001, type=float,
                   help='Alpha parameter controlling sensitivity to plate-like structures. \n'
                        'The higher `alpha` the less likely we are to label flat structures as tubes.')
    p.add_argument('--beta', default=1.0, type=float,
                   help='Beta parameter controlling sensitivity to locally-isotropic structures (blobs).\n'
                        'The higher `beta` the less likely we are to label blobs as tubes.')
    p.add_argument('--gamma', type=float,
                   help='Correction constant that adjusts the sensitivity to areas\n'
                        'of high variance/texture/structure. By default, half of the\n'
                        'maximum Hessian norm.')
    p.add_argument('--scale_range', nargs=2, type=float, default=[1, 10],
                   help='Range of sigma used. [%(default)s]')
    p.add_argument('--n_scales', type=int, default=10,
                   help='Number of scales. Must be greater than 1. [%(default)s]')
    p.add_argument('--use_skimage', action='store_true',
                   help='Use scikit-image implementation for Frangi filters.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_im = nib.load(args.in_image)
    in_data = in_im.get_fdata()

    scales = np.linspace(args.scale_range[0], args.scale_range[1], args.n_scales)

    if args.use_skimage:
        prob, direction, best_scales = frangi_skimage(in_data, sigmas=scales, alpha=args.alpha,
                                         beta=args.beta, gamma=args.gamma,
                                         black_ridges=False)
        nib.save(nib.Nifti1Image(best_scales.astype(np.float32), in_im.affine),
                 f'{args.out_prefix}_scales.nii.gz')
    else:
        prob, direction = frangi_foa3d(in_data, scales, args.alpha, args.beta, args.gamma)

    # scale by vesselness probability
    direction = direction * prob[..., None]

    # Generate RGB map
    rgb = np.abs(direction) * 255

    nib.save(nib.Nifti1Image(direction.astype(np.float32), in_im.affine),
             f'{args.out_prefix}_direction.nii.gz')
    nib.save(nib.Nifti1Image(rgb.astype(np.uint8), in_im.affine),
             f'{args.out_prefix}rgb.nii.gz')
    nib.save(nib.Nifti1Image(prob.astype(np.float32), in_im.affine),
             f'{args.out_prefix}_prob.nii.gz')


if __name__ == '__main__':
    main()
