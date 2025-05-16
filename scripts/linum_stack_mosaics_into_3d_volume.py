#!/usr/bin/env python3
import argparse
from scipy.ndimage import gaussian_filter1d

from linumpy.io.zarr import read_omezarr, save_zarr
from linumpy.utils_images import apply_xy_shift

import zarr
import dask.array as da
import re

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
    p.add_argument('--slicing_interval', type=float, default=0.2,
                   help='Interval between slices in mm. [%(default)s]')
    p.add_argument('--start_index', type=int,
                   help='Start index for each volume. If None, will be detected automatically.')
    return p


def compute_volume_shape(mosaics_files, dx_list, dy_list, slicing_interval):

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
    volume_shape = (int(slicing_interval/res[0])*len(mosaics_files), ny, nx)
    return volume_shape, x0, y0


def get_interface_index(vol, sigma):
    mean_intensity = np.mean(vol, axis=(1,2))

    mean_intensity = gaussian_filter1d(mean_intensity, sigma)

    d1x = np.diff(mean_intensity, 1)

    # find start index
    start_index = np.argmax(d1x)
    return start_index


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

    volume_shape, x0, y0 = compute_volume_shape(mosaics_files, dx_list, dy_list,
                                                args.slicing_interval)
    mosaics_depth = int(args.slicing_interval / res[0])
    print(f"Output volume shape: {volume_shape}")
    print(f"Mosaic depth: {mosaics_depth} voxels")

    # find index of interface for each slice
    if args.start_index is not None:
        start_index = args.start_index
    else:
        start_index = 0
        for slice_id, f in zip(slice_ids, mosaics_files):
            if slice_id == 0:
                # skip first slice to make sure we don't skip too much volume
                continue
            img, res = read_omezarr(f)
            start_index = max(start_index, get_interface_index(img, sigma=2.0))
    print(f"Interface index: {start_index}")

    store = zarr.TempStore()
    mosaic = zarr.open(store, mode="w", shape=volume_shape,
                       dtype=np.float32, chunks=(512, 512, 512))

    # Loop over the slices
    for i in tqdm(range(len(mosaics_files)), unit="slice", desc="Stacking slices"):
        # Load the slice
        f = mosaics_files[i]
        z = slice_ids[i]

        img, res = read_omezarr(f)
        img = img[start_index:start_index + mosaics_depth]

        # Get the shift values for the slice
        if i == 0:
            dx = x0
            dy = y0
        else:
            dx = np.cumsum(dx_list)[i - 1] + x0
            dy = np.cumsum(dy_list)[i - 1] + y0

        # Apply the shift
        img = apply_xy_shift(img, mosaic[:mosaics_depth, :, :], dy, dx)

        # Add the slice to the volume
        mosaic[z*mosaics_depth:(z+1)*mosaics_depth, :, :] = img

        del img

    dask_arr = da.from_zarr(mosaic)
    save_zarr(dask_arr, args.out_stack, scales=res,
              chunks=(512, 512, 512), n_levels=3)
    print(f"Output volume saved to {args.out_stack}")



if __name__ == '__main__':
    main()
