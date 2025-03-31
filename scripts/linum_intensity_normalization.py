#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Normalize the intensity in a given nifty image"""

import nibabel as nib
import numpy as np
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_image",
                   help="Full path to the input nifti volume.")
    p.add_argument("output_image",
                   help="Full path to the output volume")
    p.add_argument('--rescale', action='store_true',
                   help='Rescale intensities between 0-1.')
    return p

def main() :
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()
    input_path = Path(args.input_image)
    output_path = Path(args.output_image)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Load the image and create empty output array
    img = nib.load(input_path)
    array = img.get_fdata()
    dim = np.shape(array)
    output_array = np.zeros(dim, dtype=np.float32)

    # The intensity of each slice is normalized
    for z in tqdm(range(dim[2])):
        # Calculate the minimum and the 99th percentile values of the slice
        min_intensity = np.min(array[:,:,z])
        percentile_99 = np.percentile(array[:,:,z], 99)
        # Perform intensity normalization on the slice
        normalized_array = np.clip(array[:,:,z], min_intensity, percentile_99)
        if percentile_99 - min_intensity > 0:
            normalized_array = (normalized_array - min_intensity) / (percentile_99 - min_intensity) * 255
        # Save the normalized slice in the output array
        output_array[:,:,z] = normalized_array

    if args.rescale:
        output_array -= output_array.min()
        output_array = output_array / output_array.max()

    # Save the image
    img_normalized = nib.Nifti1Image(output_array, img.affine)
    nib.save(img_normalized, output_path)

if __name__ == "__main__":
    main()