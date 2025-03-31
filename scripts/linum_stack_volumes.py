#!/usr/bin/env python3
import argparse
from skimage.registration import optical_flow_tvl1, phase_cross_correlation
from skimage.transform import warp
from scipy.ndimage import gaussian_filter, binary_erosion
from scipy.signal import convolve

from linumpy.io.zarr import read_omezarr, save_zarr

import zarr
import dask.array as da

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_fixed')
    p.add_argument('in_moving')
    p.add_argument('in_fixed_mask')
    p.add_argument('in_moving_mask')
    p.add_argument('out_stitch')
    return p


def compute_xy_grad_magnitude(image, sigma=3.0):
    n_samples = int(6.0*sigma)
    if n_samples % 2 == 0:
        n_samples += 1
    x = np.linspace(-3.0, 3.0, n_samples)
    filter_1d = -2.0*x*np.exp(-x**2)
    dx = convolve(image, filter_1d.reshape((1, -1, 1)), mode='same')
    dy = convolve(image, filter_1d.reshape((1, 1, -1)), mode='same')
    return np.sqrt(dx**2 + dy**2)


def make_common_shape(vol_fixed, vol_moving):
    common_shape = (max(vol_fixed.shape[0], vol_moving.shape[0]),
                    max(vol_fixed.shape[1], vol_moving.shape[1]),
                    max(vol_fixed.shape[2], vol_moving.shape[2]))
    
    vol_moving_common_shape = np.zeros(common_shape)
    vol_moving_common_shape[:vol_moving.shape[0],
                            :vol_moving.shape[1],
                            :vol_moving.shape[2]] = vol_moving
    vol_fixed_common_shape = np.zeros(common_shape)
    vol_fixed_common_shape[:vol_fixed.shape[0],
                           :vol_fixed.shape[1],
                           :vol_fixed.shape[2]] = vol_fixed
    return vol_fixed_common_shape, vol_moving_common_shape


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol_fixed, res_fixed = read_omezarr(args.in_fixed)
    vol_moving, _ = read_omezarr(args.in_moving)

    mask_fixed, _ = read_omezarr(args.in_fixed_mask)
    mask_moving, _ = read_omezarr(args.in_moving_mask)

    vol_fixed, vol_moving = make_common_shape(vol_fixed, vol_moving)
    mask_fixed, mask_moving = make_common_shape(mask_fixed, mask_moving)

    aip_fixed = np.mean(vol_fixed, axis=0)
    aip_moving = np.mean(vol_moving, axis=0)

    u, v = optical_flow_tvl1(aip_fixed, aip_moving)
    nr, nc = aip_fixed.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

    # Warp each slice of moving volume
    moving_warped = vol_moving
    moving_mask_warped = mask_moving
#    for i in tqdm(range(moving_warped.shape[0])):
#        moving_warped[i] = warp(vol_moving[i],
#                                np.array([row_coords + v, col_coords + u]), mode='edge')
#        moving_mask_warped[i] = warp(mask_moving[i],
#                                     np.array([row_coords + v, col_coords + u]), mode='edge')

    # Compute gradient of both images
    # moving_grad_mag = compute_xy_grad_magnitude(moving_warped)
    # fixed_grad_mag = compute_xy_grad_magnitude(vol_fixed)

    fixed_proc = gaussian_filter(vol_fixed, sigma=(0.0, 5.0, 5.0))
    moving_proc = gaussian_filter(moving_warped, sigma=(0.0, 5.0, 5.0))

    # Compute correlation between gradients for varying Z offset
    best_z = 0
    best_corr = 0.0
    crop_fixed = 30
    fixed_compare = fixed_proc[crop_fixed]
    for z in tqdm(range(moving_proc.shape[0])):
        moving_compare = moving_proc[z]
        corr = np.sum(moving_compare*fixed_compare)
        if corr > best_corr:
            best_corr = corr
            best_z = z

    stitch_shape = (crop_fixed - best_z + vol_moving.shape[0],
                    vol_moving.shape[1], vol_moving.shape[2])
    temp_store = zarr.TempStore()
    vol_stitch = zarr.open(temp_store, mode="w",
                           shape=stitch_shape,
                           dtype=np.float32)

    vol_stitch[:crop_fixed] = vol_fixed[:crop_fixed]
    vol_stitch[crop_fixed:] = moving_warped[best_z:]

    dask_arr = da.from_zarr(vol_stitch)
    save_zarr(dask_arr, args.out_stitch, res_fixed)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(fixed_proc[crop_fixed], cmap='gray')
    axes[1].imshow(moving_proc[best_z], cmap='gray')
    axes[1].set_title(f'Best z is {best_z}')

    plt.show()


if __name__ == '__main__':
    main()
