#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Estimate orientations from grayscale image using steerable 4th order derivative
of Gaussian quadrature filters [1-2]. Takes a grayscale nifti image as input
and outputs a SH coefficients image in OME-Zarr format. The filte response is
transformed into an orientation distribution function by means of Funk-Radon
transform.
"""
import argparse
import logging
import os
import shutil

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
import dask.array as da
from dipy.data import SPHERE_FILES

from linumpy.feature.orientation import Steerable4thOrderGaussianQuadratureFilter
from linumpy.io.zarr import save_omezarr


EPILOG="""
[1] Freeman and Adelson, "The design and use of steerable filters",
    1991, IEEE Transactions on Pattern Analysis and Machine Intelligence.
[2] Derpanis and Gryn, "Three-dimensional Nth derivative of gaussian
    steerable filters", 2005, IEEE International conference on image
    processing
"""


SH_BASES = {
    'descoteaux07_legacy': ('descoteaux07', True),
    'tournier07_legacy': ('tournier07', True),
    'descoteaux07': ('descoteaux07', False),
    'tournier07': ('tournier07', False)
}


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input image (.nii or .nii.gz).')
    p.add_argument('out_image',
                   help='Output SH image (.ome.zarr or .nii.gz).')

    p.add_argument('--halfwidth', type=int, default=5,
                   help='Window half-width [%(default)s].')
    p.add_argument('--truncate', type=float, default=5.0,
                   help='Truncation value for Gaussian kernel\n'
                        '(samples are drawn between [-truncate, +truncate]) [%(default)s].')
    p.add_argument('--sh_order', default=6, type=int,
                   help='SH maximum order [%(default)s].')
    p.add_argument('--sphere_name', choices=SPHERE_FILES.keys(), default='repulsion100',
                   help='DIPY sphere defining the directions for which the filter is evaluated. [%(default)s]')
    p.add_argument('--padding_mode', choices=['reflect', 'constant'], default='reflect',
                   help='Padding mode for convolution operation [%(default)s].')
    p.add_argument('--chunks', nargs=4, type=int, default=(64, 64, 64, 64),
                   help='Chunk shape for processing and saving data [%(default)s].')
    p.add_argument('--out_quadrature_response',
                   help='Optional output path for the quadrature filter response (before Funk-Radon transform). Should be .ome.zarr or .nii.gz.')
    p.add_argument('--sh_order_max', type=int, default=6,
                   help='SH order for hist-FOD. [%(default)s]')
    p.add_argument('--sh_basis', choices=SH_BASES.keys(), default='tournier07',
                   help='SH basis for hist-FOD. [%(default)s]')
    p.add_argument('--prefilter', action='store_true',
                   help='Apply Gaussian pre-filtering to the input image before'
                        ' computing the steerable filter response.')
    
    p.add_argument('--processes', type=int, default=1,
                   help='Number of processes used by the program [%(default)s].')
    p.add_argument('-f', action='store_true', dest='overwrite',
                   help='Force overwriting of output files.')
    p.add_argument('-v', dest='verbose', action='store_true',
                   help='Verbose output.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel('INFO')

    # assert output exists
    if os.path.exists(args.out_image):
        if not args.overwrite:
            parser.error(f'File {args.out_image} already exists. '
                         'Use -f to overwrite.')
        elif ".ome.zarr" in args.out_image:
            shutil.rmtree(args.out_image)
        else:
            os.remove(args.out_image)

    # create intermediary directories
    head, _ = os.path.split(args.out_image)
    if head != '' and not os.path.exists(head):
        os.makedirs(head)

    # TODO: Input can't be nifti for very big images
    image = nib.load(args.in_image)

    # (x, y, z) voxel size in mm
    voxel_sizes = image.header.get_zooms()[:3]
    data = image.get_fdata()

    # normalize to avoid overflow
    data -= data.min()
    data /= data.max()

    if args.prefilter:
        sigma = args.halfwidth / args.truncate / np.sqrt(2.0)
        logging.info('Applying Gaussian pre-filtering with sigma=%.2f...', sigma)
        data = gaussian_filter(data, sigma, mode=args.padding_mode)
        # (re)normalize to avoid overflow
        data -= data.min()
        data /= data.max()

    sh_basis, is_legacy = SH_BASES[args.sh_basis]

    steerable_filter = Steerable4thOrderGaussianQuadratureFilter(
        data, args.halfwidth, args.truncate, args.sphere_name,
        args.sh_order, sh_basis, is_legacy, args.padding_mode,
        args.chunks, args.processes)
    odf_sh, quadrature_sh = steerable_filter.compute_odf_sh()

    if '.ome.zarr' in args.out_image:
        # Swap axes to follow (c, z, y, x) ordering expected for OME-Zarr
        odf_sh = da.moveaxis(da.from_array(odf_sh), (3, 2, 1, 0), (0, 1, 2, 3))
        save_omezarr(odf_sh, args.out_image, voxel_size=(1,) + voxel_sizes[::-1],
                    chunks=args.chunks[::-1], overwrite=args.overwrite)
    else:
        # Save as nifti (x, y, z, c)
        nib.save(nib.Nifti1Image(odf_sh[:].astype(np.float32), image.affine), args.out_image)

    if args.out_quadrature_response is not None:
        if '.ome.zarr' in args.out_quadrature_response:
            quadrature_sh = da.moveaxis(da.from_array(quadrature_sh), (3, 2, 1, 0), (0, 1, 2, 3))
            save_omezarr(quadrature_sh, args.out_quadrature_response, voxel_size=(1,) + voxel_sizes[::-1],
                        chunks=args.chunks[::-1], overwrite=args.overwrite)
        else:
            nib.save(nib.Nifti1Image(quadrature_sh[:].astype(np.float32), image.affine), args.out_quadrature_response)

    # manually cleanup zarr tempstore
    shutil.rmtree(steerable_filter.zarr_store.root)


if __name__ == '__main__':
    main()
