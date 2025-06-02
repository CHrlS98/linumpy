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
    p.add_argument("in_zarr",
                   help="Full path to a zarr file.")
    p.add_argument("out_figure",
                   help="Full path to the output figure")
    p.add_argument('--z_slice', type=int,
                   help='Slice index along first axis.')
    p.add_argument('--x_slice', type=int,
                   help='Slice index along the second axis.')
    p.add_argument('--y_slice', type=int,
                   help='Slice index along the last axis.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image, res = read_omezarr(args.in_zarr)

    z_slice = args.z_slice or image.shape[0]//2
    x_slice = args.x_slice or image.shape[1]//2
    y_slice = args.y_slice or image.shape[2]//2

    image_z = image[z_slice, :, :].T
    image_x = image[:, x_slice, :]
    image_x = image_x[::-1, ::-1]
    image_y = image[:, :, y_slice]
    image_y = image_y[::-1]

    width_ratio = [i.shape[1] for i in (image_z, image_x, image_y)]

    fig, ax = plt.subplots(1, 3, width_ratios=width_ratio)
    fig.set_size_inches(24, 10)
    fig.set_dpi(512)
    ax[0].imshow(image_z, cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
    ax[1].imshow(image_x, cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
    ax[2].imshow(image_y, cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
    for i in range(3):
        ax[i].set_axis_off()
    fig.tight_layout()
    fig.savefig(args.out_figure)


if __name__ == '__main__':
    main()