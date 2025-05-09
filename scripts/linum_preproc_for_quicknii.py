#!/usr/bin/env python3
"""
Prepare 2.5D images for manual registration using QuickNII:
https://quicknii.readthedocs.io/en/latest/index.html.
"""
import argparse
import os
import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.morphology import disk
from skimage.measure import label
from scipy.ndimage import\
    binary_fill_holes, binary_dilation,\
    gaussian_filter, center_of_mass
import zarr


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_zarr',
                   help='Input zarr directory.')
    p.add_argument('out_directory',
                   help='Output directory containing PNG images.')
    p.add_argument('--prefix', default='slice_',
                   help='Output images prefix. [%(default)s]')
    p.add_argument('--shape_resize', type=int, default=1920,
                   help='Length of longest edge. [%(default)s]')
    p.add_argument('--disable_betcrop', action='store_true',
                   help='Disable brain extraction and cropping.')
    return p


def process_slice_otsu(image, sigma=4.0, epsilon=1E-6,
                       disk_radius=10.0, n_iters_connect_regions=3,
                       win_size=100):
    zeros = image < epsilon
    image = gaussian_filter(image, sigma)
    hist, bin_edges = np.histogram(image[~zeros])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    otsu = threshold_otsu(hist=(hist, bin_centers))
    mask = image > otsu

    # remove holes
    mask = binary_fill_holes(mask, structure=disk(disk_radius))

    # try to connect the different brain regions
    mask = binary_dilation(mask, structure=disk(disk_radius),
                           iterations=n_iters_connect_regions)
    
    r_center, c_center = center_of_mass(mask)
    r_center, c_center = int(r_center), int(c_center)
    labels = label(mask, background=0)
    sorted_labels = np.argsort(
        np.bincount(
            labels[r_center - win_size[0]//2:r_center + win_size[0]//2,
                   c_center - win_size[1]//2:c_center + win_size[1]//2]
            .flatten()
        )
    )

    # barycenter should fall inside the brain.
    # if the most popular class is background,
    # take second most popular class.
    brain_label = sorted_labels[-1]
    if brain_label == 0 and len(sorted_labels) > 1:
        brain_label = sorted_labels[-2]
    mask = labels == brain_label

    # dilate mask to make sure we cover the hole brain
    mask = binary_dilation(mask, structure=disk(disk_radius))
    return mask


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    in_zarr = zarr.open(args.in_zarr, mode='r')
    in_min = np.min(in_zarr[:])
    in_max = np.max(in_zarr[:])

    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    out_prefix = os.path.join(args.out_directory, args.prefix)
    print(out_prefix)

    # brain voxels are found by a majority vote of 
    win_size = (int(0.2*in_zarr.shape[1]), int(0.2*in_zarr.shape[2]))
    for it in range(in_zarr.shape[0]):
        image = in_zarr[it]
        if not args.disable_betcrop:
            mask = process_slice_otsu(image, sigma=4.0, win_size=win_size)
            nonzero_rows, nonzero_cols = np.nonzero(mask)
            row_min, row_max = np.min(nonzero_rows), np.max(nonzero_rows)
            col_min, col_max = np.min(nonzero_cols), np.max(nonzero_cols)
            image = image[row_min:row_max, col_min:col_max]
        if image.shape[0] > image.shape[1]:
            scaling_axes = np.asarray(image.shape) / image.shape[0]
        else:
            scaling_axes = np.asarray(image.shape) / image.shape[1]
        resize_shape = scaling_axes * args.shape_resize
        image = resize(image, resize_shape, anti_aliasing=True)
        image = (image - in_min) / (in_max - in_min) * 255
        sitk.WriteImage(sitk.GetImageFromArray(image.astype(np.uint8)),
                        f'{out_prefix}{it:03d}.PNG')


if __name__ == '__main__':
    main()
