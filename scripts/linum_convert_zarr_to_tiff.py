#!/usr/bin/env python3
import argparse
import zarr
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_zarr')
    p.add_argument('out_directory')
    p.add_argument('--prefix', default='slice')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_zarr = zarr.open(args.in_zarr, mode='r')
    if not os.path.exists(args.out_directory):
        os.mkdir(args.out_directory)

    for zi in tqdm(range(in_zarr.shape[0])):
        im = in_zarr[zi]
        sitk.WriteImage(sitk.GetImageFromArray(im.astype(np.float32)),
                        os.path.join(args.out_directory, f'{args.prefix}_{zi}.tiff'),
                        useCompression=False, imageIO='TIFFImageIO')


if __name__ == '__main__':
    main()
