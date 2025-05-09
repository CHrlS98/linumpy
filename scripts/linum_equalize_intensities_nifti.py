#!/usr/bin/env python3
"""
Equalize intensities of 2.5D volume.

The script consists of 3 steps:
    1. Masking of voxels outside the brain mask.
    2. Histogram equalization of individual 2D slices.
    3. Rescaling of images such that the average value inside GM voxels is
       constant across all slices.
"""
import nibabel as nib
import numpy as np
import argparse
from skimage.exposure import equalize_adapthist
from linumpy.utils.coordinates import AXIS_NAME_TO_INDEX
import matplotlib.pyplot as plt
from tqdm import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_nifti",
                   help="Path to the input NIfTI file.")
    p.add_argument("in_brain_mask",
                   help="Path to the input brain mask.")
    p.add_argument("in_mask_gm_roi",
                   help="Path to the input GM mask NIfTI file.")
    p.add_argument("out_nifti",
                   help="Path to the output NIfTI file.")
    p.add_argument('--axis', choices=['sagittal', 'coronal', 'axial'],
                   default='sagittal',
                   help='Axis onto which alignment is done.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = nib.load(args.in_nifti)
    image_data = in_image.get_fdata()

    mask = nib.load(args.in_brain_mask).get_fdata().astype(bool)
    gm_roi = nib.load(args.in_mask_gm_roi).get_fdata().astype(bool)

    image_data = np.swapaxes(image_data, 0, AXIS_NAME_TO_INDEX[args.axis])
    image_data_histeq = np.zeros_like(image_data)
    for i in tqdm(range(image_data.shape[0])):
        image = image_data[i].copy()
        image[~mask[i]] = 0
        image -= image.min()
        image /= image.max()
        image_data_histeq[i] = equalize_adapthist(image, nbins=512, clip_limit=0.01)

    # Equalize the image intensities based on the value of the GM ROI
    target_intensities = np.mean(image_data_histeq, axis=(1, 2), where=gm_roi)
    image_data_histeq -= target_intensities[:, None, None]
    image_data_histeq += np.mean(target_intensities)  # all GM inside GM ROI has the same value on average
    image_data_histeq -= np.min(image_data_histeq[mask])
    image_data_histeq /= image_data_histeq.max()

    image_data_histeq = np.swapaxes(image_data_histeq, 0, AXIS_NAME_TO_INDEX[args.axis])
    image_data_histeq[~mask] = 0
    out_image = nib.Nifti1Image(image_data_histeq, in_image.affine)
    nib.save(out_image, args.out_nifti)


if __name__ == "__main__":
    main()