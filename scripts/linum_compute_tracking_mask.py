#!/usr/bin/env python3
"""

"""
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import grey_dilation, binary_erosion, map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import SimpleITK as sitk


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image',
                   help='Input weights from hist-fod.')
    p.add_argument('out_segment', help='Output segmentation.')
    p.add_argument('reference',
                   help='Reference. Segmentation is reshaped to this size and header is copied.')
    p.add_argument('--screenshot', default='screenshot.png',
                   help="Output screenshot file.")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    oct_im = nib.load(args.in_image)
    oct = oct_im.get_fdata()

    fig ,ax = plt.subplots(2, 3)
    fig.set_size_inches(18, 12)
    ax[0, 0].imshow(oct[444, :, :].T, cmap='magma', vmin=oct.min(), vmax=oct.max(), origin='lower')
    ax[0, 1].imshow(oct[:, 284, :].T, cmap='magma', vmin=oct.min(), vmax=oct.max(), origin='lower')
    ax[0, 2].imshow(oct[:, :, 102].T, cmap='magma', vmin=oct.min(), vmax=oct.max(), origin='lower')

    oct = grey_dilation(oct, size=5)

    mask = oct < 0.6
    mask = binary_erosion(mask, iterations=3)
    oct = np.ma.masked_array(oct, mask)

    ax[1, 0].imshow(oct[444, :, :].T, cmap='magma', vmin=oct.min(), vmax=oct.max(), origin='lower')
    ax[1, 1].imshow(oct[:, 284, :].T, cmap='magma', vmin=oct.min(), vmax=oct.max(), origin='lower')
    ax[1, 2].imshow(oct[:, :, 102].T, cmap='magma', vmin=oct.min(), vmax=oct.max(), origin='lower')

    fig.tight_layout()
    fig.savefig(args.screenshot)

    mask = ~mask

    reference = nib.load(args.reference)
    out_shape = reference.shape
    coordinates = np.stack(np.meshgrid(np.linspace(0, oct.shape[0], out_shape[0]),
                                       np.linspace(0, oct.shape[1], out_shape[1]),
                                       np.linspace(0, oct.shape[2], out_shape[2]),
                                       indexing='ij'), axis=0)
    resampled_mask = map_coordinates(mask, coordinates, order=0) > 0.0
    resampled_mask[0, :, :] = resampled_mask[-1, :, :] = 0
    resampled_mask[:, 0, :] = resampled_mask[:, -1, :] = 0
    resampled_mask[:, :, 0] = resampled_mask[:, :, -1] = 0

    nib.save(nib.Nifti1Image(resampled_mask.astype(np.uint8), reference.affine),
             args.out_segment)


if __name__ == '__main__':
    main()
