#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib
from skimage.restoration import denoise_bilateral
from linumpy.utils.coordinates import AXIS_NAME_TO_INDEX
from tqdm import tqdm

def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image')
    p.add_argument('out_image')
    p.add_argument('--sigma', type=float, default=3.0)
    p.add_argument('--axis', choices=AXIS_NAME_TO_INDEX.keys(), default='axial')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = nib.load(args.in_image)
    axis = AXIS_NAME_TO_INDEX[args.axis]
    data = in_image.get_fdata()
    data = np.swapaxes(data, 0, axis)
    for z in tqdm(range(data.shape[0])):
        data[z] = denoise_bilateral(data[z], sigma_spatial=args.sigma, sigma_color=0.2)
        data = (data * 10.0).astype(int) / 10.0

    data = np.swapaxes(data, axis, 0)
    nib.save(nib.Nifti1Image(data, in_image.affine), args.out_image)


if __name__ == '__main__':
    main()
