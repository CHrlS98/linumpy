#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Fiber orientations analysis using Frangi filters as in [1].
"""
import argparse
import nibabel as nib
import numpy as np

import os
from linumpy.feature.foa3d import frangi_filter as frangi_foa3d
from linumpy.utils.io import assert_output_exists, add_overwrite_arg


EPILOG="""
[1] Sorelli et al, 2023, "Fiber enhancement and 3D orientation analysis in label-free
    two-photon fluorescence microscopy", Scientific Reports (2023) 13:4160
"""

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image', help='Input nifti image.')
    p.add_argument('out_direction', help='Output direction nifti image.')
    p.add_argument('out_probability', help='Output probability nifti image.')
    p.add_argument('--alpha', default=0.5, type=float,
                   help='Alpha parameter controlling sensitivity to plate-like structures. \n'
                        'The higher `alpha` the less likely we are to label flat structures as tubes. [%(default)s]')
    p.add_argument('--beta', default=0.5, type=float,
                   help='Beta parameter controlling sensitivity to locally-isotropic structures (blobs).\n'
                        'The higher `beta` the less likely we are to label blobs as tubes. [%(default)s]')
    p.add_argument('--gamma', type=float,
                   help='Correction constant that adjusts the sensitivity to areas\n'
                        'of high variance/texture/structure. By default, half of the\n'
                        'maximum Hessian norm.')
    p.add_argument('--sigma', nargs='+', type=float, default=1.0,
                   help='Sigmas used. Can be a single value.[%(default)s]')
    p.add_argument('--vesselness_threshold', type=float, default=0,
                   help='Vesselness threshold for accepting a direction. [%(default)s]')
    p.add_argument('--padding_mode', default='constant',
                   choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'],
                   help='Padding mode for Frangi filter. [%(default)s]')
    p.add_argument('--padding_cval', type=float, default=0.0,
                   help='Constant value for padding. Only used if padding_mode is constant. [%(default)s]')

    p.add_argument('--save_decompose', action='store_true',
                   help='If true, saves the vesselness and direction for each scale in separate nifti files. [%(default)s]')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_im = nib.load(args.in_image)
    in_data = in_im.get_fdata().astype(np.float32)

    # Check if output files already exist and handle according to overwrite flag
    if args.save_decompose:
        prob_dirname = os.path.dirname(args.out_probability)
        dir_dirname = os.path.dirname(args.out_direction)
        prob_basename = os.path.basename(args.out_probability).replace('.nii', '').replace('.gz', '')
        dir_basename = os.path.basename(args.out_direction).replace('.nii', '').replace('.gz', '')
        for scale in np.atleast_1d(args.sigma):
            assert_output_exists(os.path.join(prob_dirname, f'{prob_basename}_scale_{scale}.nii.gz'), parser, args)
            assert_output_exists(os.path.join(dir_dirname, f'{dir_basename}_scale_{scale}.nii.gz'), parser, args)
    else:
        assert_output_exists(args.out_direction, parser, args)
        assert_output_exists(args.out_probability, parser, args)

    scales = np.atleast_1d(args.sigma)
    if args.save_decompose:
        prob_dirname = os.path.dirname(args.out_probability)
        dir_dirname = os.path.dirname(args.out_direction)
        prob_basename = os.path.basename(args.out_probability).replace('.nii', '').replace('.gz', '')
        dir_basename = os.path.basename(args.out_direction).replace('.nii', '').replace('.gz', '')
        for scale in scales:
            prob, direction = frangi_foa3d(in_data, [scale],
                                         alpha=args.alpha,
                                         beta=args.beta,
                                         gamma=args.gamma,
                                         threshold=args.vesselness_threshold,
                                         padding_mode=args.padding_mode,
                                         padding_cval=args.padding_cval)
            nib.save(nib.Nifti1Image(prob.astype(np.float32), in_im.affine), os.path.join(prob_dirname, f'{prob_basename}_scale_{scale}.nii.gz'))
            nib.save(nib.Nifti1Image(direction.astype(np.float32), in_im.affine), os.path.join(dir_dirname, f'{dir_basename}_scale_{scale}.nii.gz'))
    else:
        prob, direction = frangi_foa3d(in_data, scales,
                                    alpha=args.alpha,
                                    beta=args.beta,
                                    gamma=args.gamma,
                                    threshold=args.vesselness_threshold,
                                    padding_mode=args.padding_mode,
                                    padding_cval=args.padding_cval)

        nib.save(nib.Nifti1Image(direction.astype(np.float32), in_im.affine), args.out_direction)
        nib.save(nib.Nifti1Image(prob.astype(np.float32), in_im.affine), args.out_probability)


if __name__ == '__main__':
    main()
