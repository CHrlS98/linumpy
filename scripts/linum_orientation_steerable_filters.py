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
import dask.array as da
from dipy.data import SPHERE_FILES

from linumpy.feature.orientation import Steerable4thOrderGaussianQuadratureFilter
from linumpy.io.zarr import save_zarr


EPILOG="""
[1] Freeman and Adelson, "The design and use of steerable filters",
    1991, IEEE Transactions on Pattern Analysis and Machine Intelligence.
[2] Derpanis and Gryn, "Three-dimensional Nth derivative of gaussian
    steerable filters", 2005, IEEE International conference on image
    processing
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input image (.nii or .nii.gz).')
    p.add_argument('out_image',
                   help='Output SH image (.ome.zarr).')

    p.add_argument('--halfwidth', type=int, default=5,
                   help='Window half-width [%(default)s].')
    p.add_argument('--sh_order', default=6, type=int,
                   help='SH maximum order [%(default)s].')
    p.add_argument('--sphere_name', choices=SPHERE_FILES.keys(), default='repulsion100',
                   help='DIPY sphere defining the directions for which the filter is evaluated. [%(default)s]')
    p.add_argument('--padding_mode', choices=['reflect', 'constant'], default='reflect',
                   help='Padding mode for convolution operation [%(default)s].')
    p.add_argument('--chunks', nargs=4, type=int, default=(64, 64, 64, 64),
                   help='Chunk shape for processing and saving data [%(default)s].')
    
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
        else:
            shutil.rmtree(args.out_image)

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

    steerable_filter = Steerable4thOrderGaussianQuadratureFilter(
        data, args.halfwidth, args.sphere_name, args.sh_order,
        args.padding_mode, args.chunks, args.processes)
    odf_sh = steerable_filter.compute_odf_sh()

    # Swap axes to follow (c, z, y, x) ordering expected for OME-Zarr
    odf_sh = da.moveaxis(da.from_array(odf_sh), (3, 2, 1, 0), (0, 1, 2, 3))
    save_zarr(odf_sh, args.out_image, scale=(1,) + voxel_sizes[::-1],
              chunks=args.chunks[::-1], overwrite=args.overwrite)


if __name__ == '__main__':
    main()
