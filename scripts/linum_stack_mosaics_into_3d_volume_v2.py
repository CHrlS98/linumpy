#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from pathlib import Path
import re
import zarr
import dask.array as da
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.utils_images import apply_xy_shift


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_mosaics_dir",
                   help="Path to the directory containing the mosaics.")
    p.add_argument("in_xy_shifts",
                   help="Path to the file containing the XY shifts.")
    p.add_argument("out_stack",
                   help="Path to the output stack.")
    p.add_argument("--slicing_interval", type=float, default=0.200,
                   help="Interval between slices in mm. [%(default)s]")
    p.add_argument("--factor_extra", type=float, default=1.1,
                   help='Factor by which to increase the stacking interval. [%(default)s]')
    p.add_argument('--stacking_offset', type=float, default=0.030,
                   help='Skip this many mm at the beginning of each mosaic. [%(default)s]')
    p.add_argument("--sigma", type=float, default=0.030,
                   help="Gaussian smoothing sigma (mm) for "
                   "image boundary detection. [%(default)s]")
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
            xmax.append(shape[-2])
            ymin.append(0)
            ymax.append(shape[-1])
        else:
            dx = np.cumsum(dx_list)[i - 1]
            dy = np.cumsum(dy_list)[i - 1]
            xmin.append(-dx)
            xmax.append(-dx + shape[-2])
            ymin.append(-dy)
            ymax.append(-dy + shape[-1])

    # Get the volume shape
    x0 = min(xmin)
    y0 = min(ymin)
    x1 = max(xmax)
    y1 = max(ymax)
    nx = int((x1 - x0))
    ny = int((y1 - y0))

    # Important!!! The +1 is to make sure that the last mosaic
    # fits in the volume for the case where the best offset is always 1
    volume_shape = (mosaics_depth*len(mosaics_files) + 1, nx, ny)
    return volume_shape, x0, y0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_mosaics_dir = Path(args.in_mosaics_dir)
    mosaics_files = [p for p in in_mosaics_dir.glob('*.ome.zarr')]
    mosaics_files.sort()
    pattern = r".*z(\d+)_.*"  # the parentheses create a group containing the slice id
    slice_ids = []
    for f in mosaics_files:
        match = re.match(pattern, f.name)
        # the only group found is the slice id
        id = int(match.groups()[0])
        slice_ids.append(id)

    # Load cvs containing the shift values for each slice
    df = pd.read_csv(args.in_xy_shifts)

    # We load the shifts in mm, but we need to convert them to pixels
    # The shifts are only useful for computing the out shape
    dx_list = np.array(df["x_shift_mm"].tolist(), dtype=float)
    dy_list = np.array(df["y_shift_mm"].tolist(), dtype=float)

    # assume that the resolution is the same for all slices
    img, res = read_omezarr(mosaics_files[slice_ids[0]])

    # order (z, x, y)
    dx_list /= res[1]
    dy_list /= res[2]

    dx_cumsum = np.append([0], np.cumsum(dx_list))
    dy_cumsum = np.append([0], np.cumsum(dy_list))

    interval_vox = int(np.ceil(args.slicing_interval*args.factor_extra / res[0]))  # in voxels
    volume_shape, x0, y0 = compute_volume_shape(mosaics_files, interval_vox,
                                                dx_list, dy_list)

    offset_vox = int(args.stacking_offset / res[0])  # in voxels

    sigma_vox = args.sigma / res[0]
    xmin, xmax = None, None
    ymin, ymax = None, None
    output_store = zarr.TempStore()
    output_stack = zarr.open(output_store, mode="w", shape=volume_shape,
                             dtype=img.dtype, chunks=(256, 256, 256))
    for i, f in enumerate(tqdm(mosaics_files)):
        vol, _ = read_omezarr(f)
        vol = vol[offset_vox:]  # small offset to avoid incomplete slices at interface
        z_offset = i * interval_vox
        dx = x0 + dx_cumsum[i]
        dy = y0 + dy_cumsum[i]

        # dy and dx must be swapped because sitk inverts the order of the axes
        vol = apply_xy_shift(vol, output_stack[:, :, :], dy, dx)

        aip = np.mean(vol, axis=0)
        otsu = threshold_otsu(aip[aip > 0])
        smooth_aip = gaussian_filter(aip, sigma=sigma_vox)
        aip_mask = smooth_aip > otsu
        idx, idy = np.nonzero(aip_mask)
        curr_xmin = idx.min()
        curr_xmax = idx.max()
        curr_ymin = idy.min()
        curr_ymax = idy.max()
        xmin = curr_xmin if xmin is None else min(xmin, curr_xmin)
        xmax = curr_xmax if xmax is None else max(xmax, curr_xmax)
        ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
        ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

        depth = min(vol.shape[0], output_stack.shape[0] - z_offset)
        output_stack[z_offset:z_offset + depth, :, :] = vol[:depth]

    dask_out = da.from_zarr(output_stack)
    save_omezarr(dask_out[:, xmin:xmax, ymin:ymax],
                 args.out_stack, voxel_size=res,
                 chunks=output_stack.chunks, n_levels=5)


if __name__ == "__main__":
    main()
