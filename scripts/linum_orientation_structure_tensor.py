#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Perform structure tensor analysis on Nifti image.
"""
import argparse
from skimage.feature import structure_tensor
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
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Voxels below the threshold won\'t be considered for analysis. [%(default)s]')
    p.add_argument('--sigma', type=float, default=1.0,
                   help='Standard deviation of Gaussian windowing function. [%(default)s]')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_im = nib.load(args.in_image)
    in_data = in_im.get_fdata()

    A_00, A_01, A_02, A_11, A_12, A_22 = structure_tensor(in_data, args.sigma)

    A = np.zeros(in_data.shape + (3, 3))
    A[..., 0, 0] = A_00
    A[..., 0, 1] = A[..., 1, 0] = A_01
    A[..., 0, 2] = A[..., 2, 0] = A_02
    A[..., 1, 1] = A_11
    A[..., 1, 2] = A[..., 2, 1] = A_12
    A[..., 2, 2] = A_22

    mask = in_data > args.threshold

    evals, evecs = np.linalg.eigh(A[mask])

    evals_full = np.zeros(in_data.shape + (3,))
    evals_full[mask] = evals

    evecs_full = np.zeros(in_data.shape + (3, 3))
    evecs_full[mask] = evecs

    pdir = evecs_full[..., 0]

    nib.save(nib.Nifti1Image(pdir.astype(np.float32), in_im.affine), args.out_pdir)
    if args.out_evals is not None:
        nib.save(nib.Nifti1Image(evals_full.astype(np.float32), in_im.affine),  args.out_evals)


if __name__ == '__main__':
    main()
