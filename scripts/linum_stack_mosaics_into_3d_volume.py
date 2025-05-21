#!/usr/bin/env python3
import argparse

from linumpy.io.zarr import read_omezarr, save_zarr
from linumpy.utils_images import apply_xy_shift
from linumpy.stitching.registration import register_consecutive_3d_mosaics

import zarr
import dask.array as da
import re

from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
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
    p.add_argument('--initial_search', type=int, default=20,
                   help='Initial depth for depth matching (in voxels). [%(default)s]')
    p.add_argument('--depth_offset', type=int, default=50,
                   help='Offset from interface for each volume. [%(default)s]')
    p.add_argument('--max_allowed_overlap', type=int, default=5,
                   help='Maximum allowed overlap for the alignment. [%(default)s]')
    p.add_argument('--method', choices=['phase_cross_correlation', 'sitk_affine_2d', 'none'],
                   default='sitk_affine_2d',
                   help='Method to use for alignment. [%(default)s]')
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
            xmax.append(-dx + shape[-1])
            ymin.append(-dy)
            ymax.append(-dy + shape[-2])

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


def align_phase_cross_correlation(prev_mosaic, img, max_allowed_overlap):
    errors = []
    shifts = []
    for i in range(max_allowed_overlap):
        # if the lowest error is the one at the end of
        # previous slice, we don't need to shift anything
        px_shift, error, _ = phase_cross_correlation(
            prev_mosaic[-1, :, :], img[i, :, :],
            normalization=None, disambiguate=True
        )
        shifts.append(px_shift)
        errors.append(error)
    best_offset = np.argmin(errors)  # this one starts at 0
    px_shift = shifts[best_offset]
    img = shift(img, (0.0, px_shift[0], px_shift[1]))
    return img[best_offset:], best_offset


def align_sitk_affine_2d(prev_mosaic, img, max_allowed_overlap):
    best_offset = 0
    min_error = np.inf
    best_warp = np.zeros(img.shape)
    for i in range(max_allowed_overlap):
        warped_img, error = register_consecutive_3d_mosaics(prev_mosaic[-1, :, :], img[i:, :, :])
        if error < min_error:
            min_error = error
            best_offset = i
            best_warp[i:] = warped_img
    best_warp = best_warp[best_offset:]
    return best_warp, best_offset


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

    mosaics_depth = args.initial_search
    volume_shape, x0, y0 = compute_volume_shape(mosaics_files,
                                                mosaics_depth,
                                                dx_list, dy_list)
    print(f"Output volume shape: {volume_shape}")
    print(f"Mosaic depth: {mosaics_depth} voxels")

    start_index = args.depth_offset
    print(f"Interface index: {start_index}")

    mosaic_store = zarr.TempStore()
    mosaic = zarr.open(mosaic_store, mode="w", shape=volume_shape,
                       dtype=np.float32, chunks=(256, 256, 256))

    prev_mosaic = None
    current_z_offset = 0

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
        clip_ubound = np.percentile(img, 99.5, axis=(1, 2), keepdims=True)
        img = np.clip(img, a_min=None, a_max=clip_ubound)
        if img.max() - img.min() > 0.0:
            img /= np.max(img, axis=(1, 2), keepdims=True)

        best_offset = 0
        if prev_mosaic is not None:
            if args.method == 'phase_cross_correlation':
                img, best_offset = align_phase_cross_correlation(
                    prev_mosaic, img, args.max_allowed_overlap
                )
            elif args.method == 'sitk_affine_2d':
                img, best_offset = align_sitk_affine_2d(
                    prev_mosaic, img, args.max_allowed_overlap
                )

        print(f'Best offset is {best_offset}')

        # Add the slice to the volume
        mosaic[current_z_offset:current_z_offset + len(img) - 1, :, :] = img[:-1]

        # Save last volume of stack for registration of next slice
        prev_mosaic = img

        # update the z offset
        current_z_offset += len(img) - 1

    dask_arr = da.from_zarr(mosaic)
    save_zarr(dask_arr, args.out_stack, scales=res,
              chunks=(256, 256, 256), n_levels=3)

    print(f"Output volume saved to {args.out_stack}")


if __name__ == '__main__':
    main()
