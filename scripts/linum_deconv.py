#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack 3D mosaics on top of each other in a single 3D volume using the
transforms from `linum_estimate_transform_pairwise.py`. Expects all 3D
mosaics to be in the same space (same volume dimensions).
"""
import argparse
import re
from pathlib import Path
import numpy as np
from linumpy.io.zarr import read_omezarr, OmeZarrWriter
from linumpy.stitching.registration import apply_transform
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from skimage.exposure import match_histograms
import SimpleITK as sitk


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_mosaics_dir',
                   help='Input mosaics directory in .ome.zarr format.')
    p.add_argument('in_transforms_dir',
                   help='Input transforms directory. Each subdirectory should have the\n'
                   'same name as the corresponding mosaic file (without the .ome.zarr\n'
                   'extension) and contain a .mat transform file and .txt offsets file.')
    p.add_argument('out_stack',
                   help='Output stack in .ome.zarr format.')
    p.add_argument('--normalize', action='store_true',
                   help='Normalize slices during reconstruction.')
    return p


def get_input(mosaics_dir, transforms_dir, parser):
    # get all .ome.zarr files in in_mosaics_dir
    in_mosaics_dir = Path(mosaics_dir)
    in_transforms_dir = Path(transforms_dir)
    mosaics_files = [p for p in in_mosaics_dir.glob('*.ome.zarr')]
    pattern = r".*z(\d+)_.*"
    slice_ids = []
    for f in mosaics_files:
        foo = re.match(pattern, f.name)
        slice_id = int(foo.groups()[0])
        slice_ids.append(slice_id)

    transforms = []
    offsets = []
    mosaics_sorted = []
    slice_ids_argsort = np.argsort(slice_ids)
    first_mosaic = mosaics_files[slice_ids_argsort[0]]
    for arg_idx in slice_ids_argsort[1:]:
        f = mosaics_files[arg_idx]
        current_transform_dirname, ext = os.path.splitext(f.name)
        while not ext == '':  # remove all trailing extensions
            current_transform_dirname, ext = os.path.splitext(current_transform_dirname)
        current_transform_dir = in_transforms_dir / current_transform_dirname

        if not os.path.exists(current_transform_dir):
            parser.error(f'Transform {current_transform_dir} not found.')

        current_mat_file = list(current_transform_dir.glob('*.mat'))
        current_txt_file = list(current_transform_dir.glob('*.txt'))
        if len(current_mat_file) != 1:
            parser.error(f'Found {len(current_mat_file)} .mat file under {current_transform_dir.as_posix()}')
        current_mat_file = current_mat_file[0]
        if len(current_txt_file) > 1:
            parser.error(f'Found {len(current_txt_file)} .txt file under {current_transform_dir.as_posix()}')
        current_txt_file = current_txt_file[0]
        mosaics_sorted.append(f)
        transforms.append(sitk.ReadTransform(current_mat_file))
        offsets.append(np.loadtxt(current_txt_file))
    return first_mosaic, mosaics_sorted, transforms, np.array(offsets, dtype=int)


def normalize(vol, percentile_min=0.0, percentile_max=99.5):
    pmin = np.percentile(vol, percentile_min, axis=(1, 2))
    pmax = np.percentile(vol, percentile_max, axis=(1, 2))
    divisor = pmax - pmin
    vol = (vol - pmin[:, None, None])
    vol[divisor > 0] = vol[divisor > 0] / np.reshape(divisor[divisor > 0], (-1, 1, 1))
    return vol


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    first_mosaic, mosaics_sorted, transforms, offsets =\
        get_input(args.in_mosaics_dir, args.in_transforms_dir, parser)

    vol_a, res = read_omezarr(first_mosaic)
    vol_b, res = read_omezarr(mosaics_sorted[0])
    composite_transform = sitk.CompositeTransform(transforms[0])
    vol_b = apply_transform(vol_b[:], composite_transform)

    offsets_fixed = offsets[:, 0]
    offsets_moving = offsets[:, 1]
    end_of_vol_a = np.count_nonzero(np.sum(vol_a, axis=(1, 2)) > 0)

    overlap = end_of_vol_a - offsets_fixed[0]

    x_volume = vol_a[offsets_fixed[0]:end_of_vol_a]
    y_volume = vol_b[offsets_moving[0]:offsets_moving[0]+overlap]
    vmax = np.max(y_volume)
    print(vmax)

    xyz_shift, _, _ = phase_cross_correlation(x_volume, y_volume)
    y_volume_shift = shift(y_volume, xyz_shift)
    print(xyz_shift)

    n_items_per_row = 4
    n_rows = overlap // n_items_per_row +1
    fig_a ,axes_a = plt.subplots(n_rows, n_items_per_row, sharex=True, sharey=True)

    for ii in range(n_rows):
        for jj in range(n_items_per_row):
            axes_a[ii, jj].set_axis_off()

    for i in range(overlap):
        x_image = x_volume[i]
        y_image_shift = y_volume_shift[i]
        gain = np.sum(y_image_shift) / np.sum(x_image)
        rgb = np.zeros((x_image.shape[0], x_image.shape[1], 3))
        vmax = np.percentile(y_image_shift, 99)
        rgb[..., 0] = gain * x_image / vmax  # blurrier
        rgb[..., 1] = np.clip(y_image_shift / vmax, 0, 1)  # sharper
        gray = np.sqrt(rgb[..., 0]**2 + rgb[..., 1]**2)

        ii = i // n_items_per_row
        jj = i % n_items_per_row
        axes_a[ii, jj].imshow(gray[::-1])
        axes_a[ii, jj].set_title(f'gain: {gain:.2f}')
    fig_a.set_size_inches(12, 4)
    fig_a.tight_layout()
    fig_a.suptitle('Overlapping bottom of previous volume (red) with surface of next volume (green)')
    plt.show()


if __name__ == "__main__":
    main()
