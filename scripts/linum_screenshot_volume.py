#!/usr/bin/env python3
"""
Offline screenshot of a 3D volume. This is useful for quickly visualizing the data without
needing to open a viewer. The screenshot is taken along the three orthogonal planes
intersecting at the center of the volume by default, but the user can specify the slice
indices along each axis. Supports nifti and .ome.zarr files as input.
"""
import argparse
from linumpy.io.zarr import read_omezarr

import nibabel as nib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_data",
                   help="Full path to a zarr or nifti file.")
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

    if '.ome.zarr' in args.in_data:
        image, _ = read_omezarr(args.in_data)
    elif '.nii' in args.in_data:
        image = nib.load(args.in_data).get_fdata()
    else:
        parser.error(f'File type not supported: {args.in_data}')

    z_slice = args.z_slice if args.z_slice is not None else image.shape[0]//2
    x_slice = args.x_slice if args.x_slice is not None else image.shape[1]//2
    y_slice = args.y_slice if args.y_slice is not None else image.shape[2]//2

    image_z = image[z_slice, :, :].T
    image_x = image[:, x_slice, :]
    image_x = image_x[::-1, ::-1]
    image_y = image[:, :, y_slice]
    image_y = image_y[::-1]

    width_ratio = [i.shape[1] for i in (image_z, image_x, image_y)]

    allvals = np.concatenate([image_x.flatten(), image_y.flatten(), image_z.flatten()])
    vmin = np.min(allvals)
    vmax = np.percentile(allvals, 99.9)
    fig, ax = plt.subplots(1, 3, width_ratios=width_ratio)
    fig.set_size_inches(24, 10)
    fig.set_dpi(512)
    ax[0].imshow(image_z, cmap='magma', origin='lower', vmin=vmin, vmax=vmax)
    ax[1].imshow(image_x, cmap='magma', origin='lower', vmin=vmin, vmax=vmax)
    ax[2].imshow(image_y, cmap='magma', origin='lower', vmin=vmin, vmax=vmax)
    for i in range(3):
        ax[i].set_axis_off()
    fig.tight_layout()
    fig.savefig(args.out_figure)


if __name__ == '__main__':
    main()