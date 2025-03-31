#!/usr/bin/env python3
import argparse
import zarr
import numpy as np
from skimage.transform import resize_local_mean

def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_zarr')
    p.add_argument('out_zarr')
    p.add_argument('--out_shape', type=int)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_zarr = zarr.open(args.in_zarr, mode='r')
    out_store = zarr.DirectoryStore(args.out_zarr)
    out_zarr = zarr.open(out_store, mode='w')
    for i in range(in_zarr.shape[0]):
        arr = in_zarr[i]
        if args.out_shape is not None:
            arr = resize_local_mean(arr, (args.out_shape, args.out_shape),
                                    preserve_range=True)
        out_zarr.array(f'{i}', arr[None, :, :])


if __name__ == '__main__':
    main()
