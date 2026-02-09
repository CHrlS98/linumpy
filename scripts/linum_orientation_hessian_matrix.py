#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Perform multiscale 2D Hessian matrix analysis on Nifti image. Frangi definitions
of blobness and vesselness are used.
"""
import argparse
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import nibabel as nib
import numpy as np
from linumpy.utils.io import assert_output_exists, add_overwrite_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input nifti image.')
    p.add_argument('out_pdir',
                   help='Output principal direction nifti image.')
    p.add_argument('out_vesselness',
                   help='Output vesselness nifti image.')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Voxels below the threshold won\'t be considered for analysis. [%(default)s]')
    p.add_argument('--sigma', type=float, nargs='+', default=[1.0],
                   help='Standard deviation of Gaussian derivative. [%(default)s]')
    p.add_argument('--prefilter_sigma', type=float, default=1.0,
                   help='Standard deviation of smoothing filter. [%(default)s]')
    p.add_argument('--axes', nargs=2, type=int, default=[1, 2],
                   help='Axes along which to compute structure tensor. [%(default)s]')
    add_overwrite_arg(p)
    return p


def direction_from_vesselness(data, sigma, axes=None, mask=None, beta=0.5):
    """
    The derivatives are scale-normalized.

    :param data: Input volume to filter
    :param sigma: Scale of pre-smoothing kernel used for estimating derivatives
    :param axes: Axes along which we compute the structure tensor
    """
    if axes is None:
        axes = np.arange(len(data.shape))
    ndims = len(axes)
    d_elems = []
    for axis in axes:
        _d = sigma * gaussian_filter1d(data, sigma, axis=axis, order=1)
        d_elems.append(_d)

    A = np.zeros(data.shape + (ndims, ndims), dtype=np.float32)
    for _di in range(ndims):
        for _dj in range(_di, ndims):
            print(f'matrix at ij=({_di},{_dj}): axis {axes[_di]} x axis {axes[_dj]}')
            _dij = sigma * gaussian_filter1d(d_elems[_di], sigma, axis=axes[_dj], order=1)
            A[..., _di, _dj] = A[..., _dj, _di] = _dij

    if mask is None:
        mask = data > 0.0  # ignore background voxels

    evals, evecs = np.linalg.eigh(A[mask])

    # sort evals by absolute value
    evals_argsort = np.argsort(np.abs(evals), axis=-1)
    evals = np.take_along_axis(evals, evals_argsort, axis=-1)
    evecs = np.take_along_axis(evecs, evals_argsort[..., None, :], axis=-1)

    blobness = np.abs(evals[..., 0]) / np.abs(evals[..., 1])
    structureness = np.linalg.norm(evals, axis=-1)

    vesselness = np.zeros(evals_argsort.shape[:-1])
    vesselness =\
        np.exp(-blobness**2/(2.0*beta**2)) *\
        (1.0-np.exp(-structureness**2/(structureness.max()**2)))
    vesselness[evals[..., 1] > 0] = 0.0

    # vesselness and principal direction back to 3D
    vesselness_3d = np.zeros_like(data)
    vesselness_3d[mask] = vesselness

    pdir_2d = np.zeros(data.shape + (ndims,))
    pdir_2d[mask] = evecs[..., 0]
    pdir_2d[vesselness_3d < 1e-8] = 0
    pdir_3d = np.zeros(data.shape + (3,), dtype=np.float32)

    for i, axis in enumerate(axes):
        # principal orientation corresponds to eigenvector
        # associated with the eigenvalue of lowest magnitude
        pdir_3d[..., axis] = pdir_2d[..., i]

    return pdir_3d, vesselness_3d


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_output_exists(args.out_pdir, parser, args)
    assert_output_exists(args.out_vesselness, parser, args)

    in_im = nib.load(args.in_image)
    in_data = in_im.get_fdata()

    axes = np.arange(len(in_data.shape))
    if args.axes is not None:
        axes = args.axes

    # prefilter data
    in_data = gaussian_filter(in_data, args.prefilter_sigma, axes=axes)

    mask = in_data > args.threshold
    pdir = np.zeros(in_data.shape + (3,), dtype=np.float32)
    vesselness = np.zeros_like(in_data, dtype=np.float32)

    for sigma in args.sigma:
        print(f'Current sigma: {sigma}')
        pdir_s, vesselness_s = direction_from_vesselness(in_data, sigma, axes, mask)
        replace_where = vesselness_s > vesselness
        pdir[replace_where] = pdir_s[replace_where]
        vesselness[replace_where] = vesselness_s[replace_where]

    # save principal direction
    nib.save(nib.Nifti1Image(pdir.astype(np.float32), in_im.affine), args.out_pdir)

    # save vesselness
    nib.save(nib.Nifti1Image(vesselness.astype(np.float32), in_im.affine),
             args.out_vesselness)


if __name__ == '__main__':
    main()
