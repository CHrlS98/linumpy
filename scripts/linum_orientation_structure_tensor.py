#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Perform structure tensor analysis on Nifti image. Can be estimated
on a subset of axes.
"""
import argparse
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input nifti image.')
    p.add_argument('out_pdir',
                   help='Output principal direction nifti image.')
    p.add_argument('--out_evals',
                   help='Output optional eigenvalues nifti image.')
    p.add_argument('--out_coherence',
                   help='Output optional coherence nifti image.')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Voxels below the threshold won\'t be considered for analysis. [%(default)s]')
    p.add_argument('--sigma', type=float, default=1.0,
                   help='Standard deviation of Gaussian derivative. [%(default)s]')
    p.add_argument('--rho', type=float, default=1.0,
                   help='Standard deviation of windowing function. [%(default)s]')
    p.add_argument('--axes', nargs='+', type=int,
                   help='Axes along which to compute structure tensor. [%(default)s]')
    return p


def structure_tensor(data, sigma, rho, axes=None):
    """
    Docstring for structure_tensor

    :param data: Input volume to filter
    :param sigma: Scale of pre-smoothing kernel used for estimating derivatives
    :param rho: Integration scale for accumulating the components of the outer product of the gradient with itself
    :param axes: Axes along which we compute the structure tensor
    """
    if axes is None:
        axes = np.arange(len(data.shape))
    ndims = len(axes)
    d_elems = []
    for axis in axes:
        _d = gaussian_filter1d(data, sigma, axis=axis, order=1)
        d_elems.append(_d)

    A = np.zeros(data.shape + (ndims, ndims), dtype=np.float32)
    for _di in range(ndims):
        for _dj in range(_di, ndims):
            _dij = d_elems[_di] * d_elems[_dj]
            _dij = gaussian_filter(_dij, rho, axes=axes)
            A[..., _di, _dj] = A[..., _dj, _di] = _dij

    return A


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_im = nib.load(args.in_image)
    in_data = in_im.get_fdata()

    axes = np.arange(len(in_data.shape))
    if args.axes is not None:
        axes = args.axes

    ndims = len(axes)
    A = structure_tensor(in_data, args.sigma, args.rho, args.axes)

    mask = in_data > args.threshold
    evals, evecs = np.linalg.eigh(A[mask])

    coherence = (evals[..., 0] - evals[..., -1])**2
    divisor = (evals[..., 0] + evals[..., -1])**2
    coherence[divisor > 0] /= divisor[divisor > 0]

    evals_full = np.zeros(in_data.shape + (ndims,))
    evals_full[mask] = evals

    coherence_full = np.zeros(in_data.shape, dtype=np.float32)
    coherence_full[mask] = coherence

    evecs_full = np.zeros(in_data.shape + (ndims, ndims))
    evecs_full[mask] = evecs

    pdir = np.zeros(in_data.shape + (3,), dtype=np.float32)
    for i, axis in enumerate(axes):
        pdir[..., axis] = evecs_full[..., i, 0] * in_data

    # save principal direction
    nib.save(nib.Nifti1Image(pdir.astype(np.float32), in_im.affine), args.out_pdir)

    # save eigenvalues
    if args.out_evals is not None:
        nib.save(nib.Nifti1Image(evals_full.astype(np.float32), in_im.affine),  args.out_evals)

    # save coherence
    if args.out_coherence is not None:
        nib.save(nib.Nifti1Image(coherence_full.astype(np.float32), in_im.affine),  args.out_coherence)


if __name__ == '__main__':
    main()
