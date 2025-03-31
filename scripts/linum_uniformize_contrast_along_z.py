#!/usr/bin/env python3
import argparse

import numpy as np

from linumpy.io.zarr import read_omezarr, save_zarr
import dask.array as da
import zarr
from tqdm import tqdm
from skimage.exposure import equalize_hist


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image')
    p.add_argument('in_mask')
    p.add_argument('out_attenuation')
    p.add_argument('--clip_min', type=float, default=2)
    p.add_argument('--clip_max', type=float, default=98)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_image)
    mask, res = read_omezarr(args.in_mask)
    tile_shape = vol.chunks

    # vermeer et al model
    temp_store = zarr.TempStore()
    vol_corr = zarr.open(temp_store, mode="w", shape=vol.shape,
                         dtype=np.float32, chunks=tile_shape)
    for i in tqdm(range(vol.shape[0])):
        temp_vol = vol[i]
        temp_mask = mask[i]
        if (temp_mask > 0).any():
            temp_vol = temp_vol / np.mean(temp_vol[temp_mask > 0])
            vol_corr[i] = temp_vol
        else:
            vol_corr[i] = 0.0

    dask_arr = da.from_zarr(vol_corr)
    save_zarr(dask_arr, args.out_attenuation)


if __name__ == '__main__':
    main()
