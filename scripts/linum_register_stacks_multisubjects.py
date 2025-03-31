#!/usr/bin/env python3
import argparse
import os
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from linumpy.utils.registration import ITKRegistration


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('fixed_directory')
    p.add_argument('moving_directory')
    p.add_argument('out_directory')
    p.add_argument('--prefix', default='stack_')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.out_directory):
        os.mkdir(args.out_directory)

    fixed_dir = Path(args.fixed_directory)
    n_images_fixed = len(tuple(fixed_dir.glob("*.tiff")))

    moving_dir = Path(args.moving_directory)
    n_images_moving = len(tuple(moving_dir.glob("*.tiff")))

    n_image_pairs = min(n_images_fixed, n_images_moving)
    fixed_indices = np.arange(0, n_image_pairs)
    moving_indices = np.arange(60, 60-n_image_pairs, -1)

    # moving_i = 47
    # fixed_i = 13
    # fixed_fname = tuple(fixed_dir.glob(f'*_{fixed_i}.tiff'))[0]
    # moving_fname = tuple(moving_dir.glob(f'*_{moving_i}.tiff'))[0]
    # fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(fixed_fname))
    # # we transpose the moving image so its initial transform is closer to the fixed image
    # moving_img = sitk.GetArrayFromImage(sitk.ReadImage(moving_fname)).T

    # out, MI, transform = ITKRegistration(fixed_img, moving_img, metric='MSQ')
    # sitk.WriteImage(out, os.path.join(args.out_directory, f'{args.prefix}_{moving_i}.tiff'))
    # return

    for fixed_i, moving_i in zip(fixed_indices, moving_indices):
        fixed_fname = tuple(fixed_dir.glob(f'*_{fixed_i}.tiff'))[0]
        moving_fname = tuple(moving_dir.glob(f'*_{moving_i}.tiff'))[0]
        print(fixed_fname, moving_fname)

        fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(fixed_fname))
        # we transpose the moving image so its initial transform is closer to the fixed image
        moving_img = sitk.GetArrayFromImage(sitk.ReadImage(moving_fname)).T

        out, MI, transform = ITKRegistration(fixed_img, moving_img, metric='MSQ')
        sitk.WriteImage(out, os.path.join(args.out_directory, f'{args.prefix}_{moving_i}.tiff'))


if __name__ == '__main__':
    main()
