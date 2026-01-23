#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Perform structure tensor analysis on Nifti image.
"""
import argparse
from scipy.ndimage import gaussian_filter1d
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
    p.add_argument('--out_wm_mask',
                   help='Output WM mask.')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Voxels below the threshold won\'t be considered for analysis. [%(default)s]')
    p.add_argument('--sigma', type=float, default=1.0,
                   help='Standard deviation of Gaussian derivative. [%(default)s]')
    p.add_argument('--axes', nargs='+', type=int,
                   help='Axes along which to compute structure tensor. [%(default)s]')
    return p


def hessian_matrix(data, sigma, axes=None):
    """
    :param data: Input volume to filter
    :param sigma: Scale of pre-smoothing kernel used for estimating derivatives
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
            print(f'matrix at ij=({_di},{_dj}): axis {axes[_di]} x axis {axes[_dj]}')
            _dij = gaussian_filter1d(d_elems[_di], sigma, axis=axes[_dj], order=1)
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
    A = hessian_matrix(in_data, args.sigma, args.axes)

    mask = in_data > args.threshold
    evals, evecs = np.linalg.eigh(A[mask])

    evals_full = np.zeros(in_data.shape + (ndims,))
    evals_full[mask] = evals

    evecs_full = np.zeros(in_data.shape + (ndims, ndims))
    evecs_full[mask] = evecs

    evals_abs = np.abs(evals_full)
    evals_abs_argmin = np.argmin(evals_abs, axis=-1, keepdims=True)

    is_wm_structure = evals_abs[..., 0] > evals_abs[..., 1]

    pdir = np.zeros(in_data.shape + (3,), dtype=np.float32)
    best_evec = np.take_along_axis(evecs_full, evals_abs_argmin[..., None], axis=-1)
    best_evec = np.reshape(best_evec, best_evec.shape[:-1])  # shape along last axis is 1

    for i, axis in enumerate(axes):
        # principal orientation corresponds to eigenvector
        # associated with the eigenvalue of lowest magnitude
        pdir[..., axis] = best_evec[..., i]

    # remove invalid directions
    pdir[np.logical_not(is_wm_structure)] = 0

    # save principal direction
    nib.save(nib.Nifti1Image(pdir.astype(np.float32), in_im.affine), args.out_pdir)

    # save eigenvalues
    if args.out_evals is not None:
        nib.save(nib.Nifti1Image(evals_full.astype(np.float32), in_im.affine),  args.out_evals)

    # save WM mask
    if args.out_wm_mask is not None:
        nib.save(nib.Nifti1Image(is_wm_structure.astype(np.uint8), in_im.affine),  args.out_wm_mask)


if __name__ == '__main__':
    main()
