#!/usr/bin/env python3
"""
Extract brain mask from a 3D volume using Gaussian filtering and thresholding.
"""
import argparse
from linumpy.io.zarr import read_omezarr, save_zarr
import zarr
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

import dask.array as da
from tqdm import tqdm
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_volume',
                   help='Full path to a zarr file.')
    p.add_argument('out_masked',
                   help='Full path to the output masked image.')
    p.add_argument('out_mask',
                   help='Full path to the output zarr file.')
    p.add_argument('--sigma', type=float, default=2.0,
                   help='Sigma for Gaussian filter. [%(default)s]')
    p.add_argument('--threshold', type=float, default=0.09,
                   help='Threshold for the mask. [%(default)s]')
    p.add_argument('--iterations', type=int, default=1,
                   help='Number of iterations for binary dilation. [%(default)s]')
    return p


def process_slice(slice_vol, sigma, threshold, iterations):
    slice_vol = gaussian_filter(slice_vol, sigma)
    mask = slice_vol > threshold
    if iterations > 0:
        mask = binary_dilation(mask, iterations=iterations)
    return mask


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if '.ome.zarr' in args.in_volume:
        vol, res = read_omezarr(args.in_volume)
    else:
        vol = zarr.open(args.in_volume, mode='r')
    chunk_size = vol.chunks

    # assumption: all slices contain foreground and background
    mask_store = zarr.TempStore()
    mask = zarr.open(mask_store, mode='w', shape=vol.shape,
                     dtype=int, chunks=chunk_size)
    masked_store = zarr.TempStore()
    masked_out = zarr.open(masked_store, mode='w', shape=vol.shape,
                          dtype=np.float32, chunks=chunk_size)

    n_chunks_x = int(np.ceil(vol.shape[1] / chunk_size[1]))
    n_chunks_y = int(np.ceil(vol.shape[2] / chunk_size[2]))

    for i in tqdm(range(n_chunks_x)):
        for j in range(n_chunks_y):
            x_lb = i*chunk_size[1]
            x_ub = min((i + 1)*chunk_size[1], vol.shape[1])
            y_lb = j*chunk_size[2]
            y_ub = min((j + 1)*chunk_size[2], vol.shape[2])
            chunk = vol[:, x_lb:x_ub, y_lb:y_ub]
            mask[:, x_lb:x_ub, y_lb:y_ub] = process_slice(chunk, args.sigma,
                                                          args.threshold,
                                                          args.iterations)
            masked_out[:, x_lb:x_ub, y_lb:y_ub] = \
                vol[:, x_lb:x_ub, y_lb:y_ub] * \
                mask[:, x_lb:x_ub, y_lb:y_ub]

    if '.ome.zarr' in args.out_mask:
        da_mask_out = da.from_zarr(mask)
        save_zarr(da_mask_out, args.out_mask,
                  scales=res, chunks=chunk_size)
        da_masked_out = da.from_zarr(masked_out)
        save_zarr(da_masked_out, args.out_masked,
                  scales=res, chunks=chunk_size)
    else:
        out_mask = zarr.open(args.out_mask, mode='w', chunks=vol.chunks)
        out_masked = zarr.open(args.out_masked, mode='w', chunks=vol.chunks)
        zarr.copy(mask, out_mask)
        zarr.copy(masked_out, out_masked)


if __name__ == '__main__':
    main()
