#!/usr/bin/env python3
"""

"""
from scipy.signal import resample
import argparse
import numpy as np
import zarr
import dask.array as da
from linumpy.io.zarr import read_omezarr, save_zarr
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr",
                   help="Full path to a zarr file.")
    p.add_argument("output_zarr",
                   help="Full path to the output zarr file")
    p.add_argument('factor', type=float,
                   help='Resampling factor.')
    p.add_argument('--keep_original_resolution', action='store_true',
                   help='Keep original resolution in the header.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image, res = read_omezarr(args.input_zarr)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(24, 6)
    fig.set_dpi(256)
    ax[0].imshow(image[:, :, image.shape[2]//2])
    ax[1].imshow(image[image.shape[0]//2, :, :])
    ax[2].imshow(image[:, image.shape[1]//2, :])
    fig.tight_layout()
    fig.savefig('fig.png')

    return
    initial_num_samples = image.shape[0]

    # TODO: handle non-integer division
    target_num_samples = round(args.factor * initial_num_samples)
    print(target_num_samples)

    store = zarr.TempStore()
    resampled_zarr = zarr.open(store, mode='w',
                               shape=(target_num_samples, image.shape[1], image.shape[2]),
                               dtype=image.dtype, chunks=image.chunks)

    for i in tqdm(range(image.shape[1])):
        for j in range(image.shape[2]):
            signal = image[:, i, j]
            resampled_signal = resample(signal, target_num_samples)
            resampled_zarr[:, i, j] = resampled_signal

    out_res = res
    if not args.keep_original_resolution:
        res_z = initial_num_samples / float(target_num_samples) * res[0]
        out_res = (res_z,) + res[1:]

    dask_out = da.from_zarr(resampled_zarr)
    save_zarr(dask_out, args.output_zarr, scales=out_res)


if __name__ == '__main__':
    main()