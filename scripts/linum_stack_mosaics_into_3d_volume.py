#!/usr/bin/env python3
import argparse

from linumpy.io.zarr import read_omezarr, save_zarr
from linumpy.utils_images import apply_xy_shift

import zarr
import dask.array as da
import re

from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_gradient_magnitude, shift
import numpy as np
import pandas as pd

from pathlib import Path

from tqdm import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_mosaics_dir',
                    help='Path to the directory containing the mosaics.')
    p.add_argument('in_xy_shifts',
                   help='Path to the file containing the XY shifts.')
    p.add_argument('out_stack',
                   help='Path to the output stack.')
    p.add_argument('--save_grad',
                   help='Optional filename for 2D gradient magnitude.')
    p.add_argument('--slicing_interval', type=float, default=200,
                   help='Interval between slices in microns. [%(default)s]')
    p.add_argument('--start_index', type=int, default=50,
                   help='Start index for each volume. [%(default)s]')

    return p


def compute_volume_shape(mosaics_files, mosaics_depth,
                         dx_list, dy_list):

    # Compute the volume shape
    xmin = []
    xmax = []
    ymin = []
    ymax = []

    for i, f in enumerate(mosaics_files):
        # Get this volume shape
        img, res = read_omezarr(f)
        shape = img.shape

        # Get the cumulative shift
        if i == 0:
            xmin.append(0)
            xmax.append(shape[-1])
            ymin.append(0)
            ymax.append(shape[-2])
        else:
            dx = np.cumsum(dx_list)[i - 1]
            dy = np.cumsum(dy_list)[i - 1]
            xmin.append(-dx)
            xmax.append(-dx + shape[1])
            ymin.append(-dy)
            ymax.append(-dy + shape[0])

    # Get the volume shape
    x0 = min(xmin)
    y0 = min(ymin)
    x1 = max(xmax)
    y1 = max(ymax)
    nx = int((x1 - x0))
    ny = int((y1 - y0))

    # TODO: Handle the case where resolution does not perfectly
    # divides the slicing interval
    volume_shape = (mosaics_depth*len(mosaics_files), ny, nx)
    return volume_shape, x0, y0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # get all .ome.zarr files in in_mosaics_dir
    in_mosaics_dir = Path(args.in_mosaics_dir)
    mosaics_files = [p for p in in_mosaics_dir.glob('*.ome.zarr')]
    mosaics_files.sort()
    pattern = r".*z(\d+)_.*"
    slice_ids = []
    for f in mosaics_files:
        foo = re.match(pattern, f.name)
        slice_ids.append(int(foo.groups()[0]))
    slice_ids = np.array(slice_ids)

    # when indexing does not start from 0, we need to shift the slice ids
    slice_ids -= np.min(slice_ids)

    # Load cvs containing the shift values for each slice
    df = pd.read_csv(args.in_xy_shifts)

    # We load the shifts in mm, but we need to convert them to pixels
    dx_list = np.array(df["x_shift_mm"].tolist())
    dy_list = np.array(df["y_shift_mm"].tolist())

    # assume that the resolution is the same for all slices
    img, res = read_omezarr(mosaics_files[0])
    dx_list /= res[-1]
    dy_list /= res[-2]

    mosaics_depth = round(args.slicing_interval / 1000.0 / res[0])
    volume_shape, x0, y0 = compute_volume_shape(mosaics_files,
                                                mosaics_depth,
                                                dx_list, dy_list)
    print(f"Output volume shape: {volume_shape}")
    print(f"Mosaic depth: {mosaics_depth} voxels")

    start_index = args.start_index
    print(f"Interface index: {start_index}")

    mosaic_store = zarr.TempStore()
    mosaic = zarr.open(mosaic_store, mode="w", shape=volume_shape,
                       dtype=np.float32, chunks=(512, 512, 512))
    
    grad_store = zarr.TempStore()
    vol_grad = zarr.open(grad_store, mode="w", shape=volume_shape,
                     dtype=np.float32, chunks=(512, 512, 512))

    prev_mosaic_bottom = np.zeros((mosaic.shape[1:]))

    errors = []
    # Loop over the slices
    for i in tqdm(range(len(mosaics_files)), unit="slice", desc="Stacking slices"):
        # Load the slice
        f = mosaics_files[i]
        z = slice_ids[i]

        img, res = read_omezarr(f)
        img = img[start_index:start_index + mosaics_depth + 1]

        # Get the shift values for the slice
        dx = x0
        dy = y0
        if i > 0:
            dx += np.cumsum(dx_list)[i - 1]
            dy += np.cumsum(dy_list)[i - 1]

        # Apply the shift as an initial alignment
        img = apply_xy_shift(img, mosaic[:mosaics_depth + 1, :, :], dy, dx)

        # Equalize intensities
        clip_ubound = np.percentile(img, 99, axis=(1, 2), keepdims=True)
        img = np.clip(img, a_min=None, a_max=clip_ubound)
        if img.max() - img.min() > 0.0:
            img /= np.max(img, axis=(1, 2), keepdims=True)

        if i > 0:
            # Register volume at depth 0 to previous volume at depth mosaics_depth + 1
            px_shift, error, _ = phase_cross_correlation(prev_mosaic_bottom, img[0, :, :],
                                                         normalization=None,
                                                         disambiguate=True)
            errors.append(error)
            img = shift(img, (0.0, px_shift[0], px_shift[1]))

        # Compute 2D norm of gradient.
        img_grad = gaussian_gradient_magnitude(img, sigma=3.0, axes=(1,2))
        if img_grad.max() - img_grad.min() > 0.0:
            img_grad /= np.max(img_grad, axis=(1, 2), keepdims=True)

        # Add the slice to the volume
        mosaic[z*mosaics_depth:(z+1)*mosaics_depth, :, :] = img[:mosaics_depth, :, :]
        vol_grad[z*mosaics_depth:(z+1)*mosaics_depth, :, :] = img_grad[:mosaics_depth, :, :]

        # Save last slice of stack for registration of next slice
        prev_mosaic_bottom = img[-1, :, :]
        del img

    dask_arr = da.from_zarr(mosaic)
    save_zarr(dask_arr, args.out_stack, scales=res,
              chunks=(512, 512, 512), n_levels=3)

    if args.save_grad:
        dask_grad = da.from_zarr(vol_grad)
        save_zarr(dask_grad, args.save_grad, scales=res,
                  chunks=(512, 512, 512), n_levels=3)
        print(f"Gradients saved to {args.save_grad}")

    print(f"Output volume saved to {args.out_stack}")

    print(f"Mean registration error is {np.mean(errors)}")


if __name__ == '__main__':
    main()
