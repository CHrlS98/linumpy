#!/usr/bin/env python3
"""
Downscale 2.5D zarr image by an integer factor.
"""
import argparse
import zarr
from tqdm import tqdm
from skimage.transform import downscale_local_mean


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_zarr',
                   help='Input zarr file to downscale.')
    p.add_argument('out_zarr',
                   help='Output downscaled zarr file.')
    p.add_argument('downscaling_factor', type=int,
                   help='Downscaling factor (integer value) [%(default)s].')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    data = zarr.open(args.in_zarr, mode='r')

    # this step to guess the output shape
    temp_image = data[0, ::args.downscaling_factor, ::args.downscaling_factor]

    out_shape = (len(data),) + temp_image.shape

    # create an empty array with output shape and fill it slice by slice
    out_arr = zarr.open(args.out_zarr, shape=out_shape)
    for i in tqdm(range(data.shape[0])):
        out_arr[i, :, :] = downscale_local_mean(data[i, :, :], args.downscaling_factor)


if __name__ == '__main__':
    main()