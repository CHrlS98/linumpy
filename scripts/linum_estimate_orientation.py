#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import shutil
from tempfile import mkdtemp
import nibabel as nib
import numpy as np
import dask.array as da
from dipy.data import get_sphere
from dipy.core.sphere import HemiSphere, Sphere
from dipy.reconst.shm import sh_to_sf_matrix, sph_harm_ind_list
from scipy.special import eval_legendre
from linumpy.feature.orientation import Steerable4thOrderGaussianQuadratureFilter
from linumpy.io.zarr import save_zarr


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input image.')
    p.add_argument('out_image',
                   help='Output SH (FRT) image.')

    p.add_argument('--mask',
                   help='Optional image used for masking the outputs.')

    p.add_argument('--out_energy',
                   help='Optional output energy as SH coefficents (not transformed).')
    p.add_argument('--out_sharpness',
                   help='Output result of Laplace-Beltrami operator.')
    p.add_argument('--R', type=int, default=5,
                   help='Window half-width [%(default)s].')
    p.add_argument('--smooth', type=float, default=0.0,
                   help='Value of lambda parameter for regularized SF to SH [%(default)s].')
    p.add_argument('--sh_order', default=6, type=int,
                   help='SH maximum order [%(default)s].')
    p.add_argument('--padding_mode', choices=['reflect', 'constant'], default='reflect',
                   help='Padding mode for convolution operation [%(default)s].')

    p.add_argument('-f', action='store_true', dest='overwrite')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = nib.load(args.in_image)
    data = image.get_fdata()

    # normalize to avoid overflow
    data -= data.min()
    data /= data.max()

    head, _ = os.path.split(args.out_image)
    if head != '' and not os.path.exists(head):
        os.makedirs(head)

    sh_order_max = args.sh_order
    _, l = sph_harm_ind_list(sh_order_max)

    FRT = np.diag(2.0*np.pi*eval_legendre(l, 0))
    L = np.diag(l**2*(l + 1)**2)

    zarr_path = "image_bank.zarr"
    steerable_filter = Steerable4thOrderGaussianQuadratureFilter(data, args.R, zarr_path,
                                                                 mode=args.padding_mode)
    print('Image bank computed...')

    sphere = HemiSphere.from_sphere(get_sphere('repulsion100'))

    # g_response = steerable_filter.compute_G_response(vertices)
    # h_response = steerable_filter.compute_H_response(vertices)
    q_response = steerable_filter.compute_quadrature_output(sphere.vertices)

    # _, b_inv = sh_to_sf_matrix(sphere, sh_order_max, smooth=args.smooth)
    # sh = np.asarray([e.dot(b_inv) for e in q_response])
    # sh = sh.dot(FRT)  # funk radon transform to invert signal
    # vertices = np.array([[1.0, 0.0, 0.0],
    #                      [0.0, 1.0, 0.0],
    #                      [0.0, 0.0, 1.0]])
    # b = sh_to_sf_matrix(Sphere(xyz=vertices), sh_order_max, return_inv=False)

    # sf = np.asarray([_sh.dot(b) for _sh in sh])
    # sf = da.moveaxis(da.from_array(sf), (3, 2, 1, 0), (0, 1, 2, 3))
    sf = da.moveaxis(da.from_array(q_response), (3, 2, 1, 0), (0, 1, 2, 3))
    save_zarr(sf, args.out_image, scale=(1, 1, 1, 1), chunks=(10, 80, 80, 80),
              overwrite=args.overwrite)

    return

    # g_response and h_response are numpy memmaps
    energy = steerable_filter.energy(sphere.vertices)
    print('Energy computed...')
    if args.mask:
        mask = nib.load(args.mask).get_fdata() > 0
        energy[~mask] = 0
    print('Max energy:', energy.max(), sphere.vertices[np.argmax(energy) % int(len(sphere.vertices))])

    _, b_inv = sh_to_sf_matrix(sphere, sh_order_max, smooth=args.smooth)
    sh = np.asarray([e.dot(b_inv) for e in energy])
    print('SH coefficients fitted...')
    print(sh.shape)

    if args.out_energy is not None:
        nib.save(nib.Nifti1Image(sh.astype(np.float32), image.affine), args.out_energy)

    sh = sh.dot(FRT)

    print('SH coefficients transformed...')
    nib.save(nib.Nifti1Image(sh, image.affine), args.out_image)

    if args.out_sharpness is not None:
        print('Computing sharpness...')
        E = np.sum((sh**2).dot(L), axis=-1)
        nib.save(nib.Nifti1Image(E.astype(np.float32), image.affine), args.out_sharpness)

    # everything is saved, delete the mmap tempdir
    shutil.rmtree(mmap_tmpdir)

    print('Mean energy is:', np.mean(energy))
    print('Mean SH0 is:', np.mean(sh[..., 0]))


if __name__ == '__main__':
    main()
