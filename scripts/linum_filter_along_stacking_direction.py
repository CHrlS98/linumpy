#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
from linumpy.io.zarr import read_omezarr, save_zarr

import zarr
import dask.array as da

from tqdm import tqdm

def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_zarr',
                   help='Input image.')
    p.add_argument('out_zarr',
                   help='Output image.')
    p.add_argument('--sigma', type=float, default=1.0,
                   help='Gaussian sigma.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image, res = read_omezarr(args.in_zarr)
    chunk_size = image.chunks

    n_chunks_x = int(np.ceil(image.shape[1] / chunk_size[1]))
    n_chunks_y = int(np.ceil(image.shape[2] / chunk_size[2]))

    store = zarr.TempStore()
    output = zarr.open(store, mode="w", shape=image.shape,
                       dtype=image.dtype, chunks=chunk_size)
    for i in tqdm(range(n_chunks_x)):
        for j in range(n_chunks_y):
            x_lb = i*chunk_size[1]
            x_ub = min((i + 1)*chunk_size[1], image.shape[1])
            y_lb = j*chunk_size[2]
            y_ub = min((j + 1)*chunk_size[2], image.shape[2])
            chunk = image[:, x_lb:x_ub, y_lb:y_ub]
            filtered_chunk = gaussian_filter1d(chunk, sigma=args.sigma, axis=0)
            output[:, x_lb:x_ub, y_lb:y_ub] = filtered_chunk

    dask_out = da.from_zarr(output)
    save_zarr(dask_out, args.out_zarr, scales=res, chunks=chunk_size, n_levels=3)


if __name__ == '__main__':
    main()
