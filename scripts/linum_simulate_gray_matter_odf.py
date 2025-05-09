#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import argparse
import nibabel as nib
import numpy as np
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.core.sphere import Sphere


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Input FODF SH coefficients.')
    p.add_argument('out_sh',
                   help='Output FODF SH coefficients.')
    p.add_argument('--mask',
                   help='Mask containing GM voxels. If None, a mask will be estimated from the data.')
    p.add_argument('--strength', default=0.001, type=float,
                   help='Strength of GM voxels.')
    p.add_argument('--mean', default=0.01, type=float,
                   help='Mean GM intensity. [%(default)s]')
    p.add_argument('--padding', nargs=6, type=int,
                   metavar=('x_before', 'x_after', 'y_before', 'y_after', 'z_before', 'z_after'),
                   help='Optional padding to apply to image.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vertices = np.array([[1.0, 0.0, 0.0]])
    _, B_inv = sh_to_sf_matrix(Sphere(xyz=vertices), sh_order_max=0, return_inv=True)

    sh_im = nib.load(args.in_sh)
    sh_data = sh_im.get_fdata()

    mask_data = sh_data[..., 0]
    if args.mask:
        mask_data = nib.load(args.in_mask).get_fdata()

    if args.padding is not None:
        pad = ((args.padding[0], args.padding[1]),
               (args.padding[2], args.padding[3]),
               (args.padding[4], args.padding[5]),
               (0, 0))
        sh_data = np.pad(sh_data, pad_width=pad)
        mask_data = np.pad(mask_data, pad_width=pad[:3])

    mask = mask_data == 0

    bg_sim = (args.mean + np.random.randn(np.count_nonzero(mask)) * args.strength) * B_inv[0, 0]
    sh_data[mask, 0] = bg_sim
    nib.save(nib.Nifti1Image(sh_data, sh_im.affine), args.out_sh)


if __name__ == '__main__':
    main()
