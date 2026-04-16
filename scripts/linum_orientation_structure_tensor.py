#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Perform structure tensor analysis on Nifti image.
"""
import argparse
from skimage.feature import structure_tensor
from scipy.ndimage import gaussian_filter
from linumpy.utils.io import assert_output_exists, add_overwrite_arg
import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input nifti image.')
    p.add_argument('out_direction',
                   help='Output principal direction nifti image.')
    p.add_argument('out_probability',
                   help='Output probability nifti image.')
    p.add_argument('--out_parameters',
                   help='Optional output nifti image to store best parameters (sigma and rho) for each voxel.')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Voxels below the threshold won\'t be considered for analysis. [%(default)s]')
    p.add_argument('--certainty_threshold', type=float, default=0.0,
                   help='certainty threshold for accepting a direction. [%(default)s]')
    p.add_argument('--sigma', type=float, nargs='+',
                   help='Standard deviation of Gaussian (scale) filter. [%(default)s]')
    p.add_argument('--rho', type=float, default=1.0, nargs='+',
                   help='Standard deviation of structure tensor window. [%(default)s]')

    p.add_argument('--padding_mode', default='constant',
                   choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'],
                   help='Padding mode for structure tensor estimation. [%(default)s]')
    p.add_argument('--padding_cval', type=float, default=0.0,
                   help='Constant value for padding. [%(default)s]')
    add_overwrite_arg(p)
    return p


def structure_tensor_wrapper(data, sigma, rho, padding_mode='constant', padding_cval=0):
    if sigma is not None and sigma > 0.0:
        data = gaussian_filter(data, sigma=sigma, mode=padding_mode, cval=padding_cval)
    a_elems = structure_tensor(data, sigma=rho, mode=padding_mode, cval=padding_cval)
    A = np.zeros(data.shape + (3, 3), dtype=np.float64)
    A[..., 0, 0] = a_elems[0]
    A[..., 0, 1] = a_elems[1]
    A[..., 0, 2] = a_elems[2]
    A[..., 1, 1] = a_elems[3]
    A[..., 1, 2] = a_elems[4]
    A[..., 2, 2] = a_elems[5]
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 2, 0] = A[..., 0, 2]
    A[..., 2, 1] = A[..., 1, 2]

    return A


def divide_nonzero(num, div):
    res = np.zeros_like(num)
    nonzero_mask = div != 0
    res[nonzero_mask] = num[nonzero_mask] / div[nonzero_mask]
    return res


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_im = nib.load(args.in_image)
    in_data = in_im.get_fdata()
    
    assert_output_exists(args.out_direction, parser, args)
    assert_output_exists(args.out_probability, parser, args)
    if args.out_parameters is not None:
        assert_output_exists(args.out_parameters, parser, args)

    mask = in_data > args.threshold
    evecs_full = np.zeros(in_data.shape + (3, 3))
    evals_full = np.zeros(in_data.shape + (3,))
    certainty = np.zeros(in_data.shape)

    # Add a certainty threshold, else all directions will be
    # considered valid for the first scale, and then only the ones with higher
    # certainty will be updated, but if the certainty is very low
    # for all scales, we might end up with noisy directions.
    best_certainty = np.full(in_data.shape, args.certainty_threshold)
    best_directions = np.zeros(in_data.shape + (3,))
    best_parameters = np.zeros(in_data.shape + (2,))  # to store best sigma and rho for each voxel

    for sigma in np.atleast_1d(args.sigma):
        for rho in args.rho:
            print('Processing sigma={} and rho={}'.format(sigma, rho))
            A = structure_tensor_wrapper(in_data, sigma, rho, args.padding_mode, args.padding_cval)
            if sigma is not None and sigma > 0.0:
                A = A * (sigma ** 2)  # scale normalization for Gaussian smoothing

            evals, evecs = np.linalg.eigh(A[mask])

            evecs_full[:] = 0  # reset to zero before filling in new values
            evals_full[:] = 0  # reset to zero before filling in new values
            certainty[:] = 0  # reset to zero before filling in new values

            # back to 3D
            evals_full[mask] = evals

            # ascending order of eigenvalues, so lambda_1 is the largest
            lambda_1 = evals_full[..., 2]
            lambda_2 = evals_full[..., 1]
            lambda_3 = evals_full[..., 0]
            certainty = divide_nonzero(lambda_2 - lambda_3, lambda_1)

            # TODO: certainty is already pretty much continuous, what we want to
            # prevent is spurious high certainty in isolated voxels due to noise
            update = certainty > best_certainty

            # back to 3D
            evecs_full[mask] = evecs
            pdir = evecs_full[..., 0]

            best_certainty[update] = certainty[update]
            best_directions[update] = pdir[update]
            best_parameters[update] = [sigma if sigma is not None else 0, rho]

    # save principal direction
    nib.save(nib.Nifti1Image(best_directions.astype(np.float32), in_im.affine), args.out_direction)

    # save certainty
    nib.save(nib.Nifti1Image(best_certainty.astype(np.float32), in_im.affine), args.out_probability)

    # save best parameters (optional)
    if args.out_parameters is not None:
        nib.save(nib.Nifti1Image(best_parameters.astype(np.float32), in_im.affine), args.out_parameters)


if __name__ == '__main__':
    main()
