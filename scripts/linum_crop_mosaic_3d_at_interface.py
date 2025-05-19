#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop a 3D mosaics volume at interface.
"""
import argparse
import numpy as np

from linumpy.io.zarr import read_omezarr, save_zarr
from scipy.ndimage import gaussian_filter1d
from skimage.filters import threshold_otsu
from scipy.signal import argrelmax
import zarr
import dask.array as da


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input',
                   help='Input mosaic volume(s) in .ome.zarr format.')
    p.add_argument('output',
                   help='Output cropped mosaic in .ome.zarr format.')
    p.add_argument('--smoothing_sigma', type=float, default=3.0,
                   help='Smoothing sigma for derivative of Gaussian. [%(default)s]')
    p.add_argument('--out_depth', type=int,
                   help='Output depth of cropped volume. If greater than volume depth,\n'
                        'volume will be zero-padded, otherwise volume will be cropped.')
    return p


def _get_interface_majority_vote(derivatives):
    interface_candidates = argrelmax(derivatives, axis=0)
    candidates_asarray = np.zeros(derivatives.shape, dtype=int)
    candidates_asarray[interface_candidates] = 1
    candidates_asarray = np.cumsum(candidates_asarray, axis=0)
    # cumsum is computed twice so that there remains only a single 1 along each z.
    candidates_asarray = np.cumsum(candidates_asarray, axis=0)

    interface = np.argwhere(candidates_asarray == 1)[:, 0]
    return np.argmax(np.bincount(interface))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.input)
    average = np.mean(vol, axis=(1, 2))
    otsu = threshold_otsu(average)
    masked_vol = np.asarray(vol)
    masked_vol[masked_vol < otsu] = 0  # enforce no derivative in the background
    derivative = np.diff(gaussian_filter1d(masked_vol,
                                           sigma=args.smoothing_sigma,
                                           axis=0), axis=0)
    interface = _get_interface_majority_vote(derivative)

    print(f"Found interface at index {interface}")

    cropped_depth = vol.shape[0] - interface
    if args.out_depth is not None:
        out_depth = args.out_depth
    else:
        out_depth = cropped_depth
    out_shape = (out_depth, vol.shape[1], vol.shape[2])
    store = zarr.TempStore()
    out_vol = zarr.open(store, mode="w", shape=out_shape,
                        dtype=np.float32, chunks=vol.chunks)
    out_vol[:min(cropped_depth, out_depth)] = vol[interface:out_depth+interface]

    dask_out = da.from_zarr(out_vol)
    save_zarr(dask_out, args.output, res, chunks=vol.chunks)


if __name__ == '__main__':
    main()
