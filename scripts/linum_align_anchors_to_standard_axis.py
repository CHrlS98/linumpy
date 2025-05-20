#!/usr/bin/env python3
"""
Shear the input NIfTI image along a specified axis to align it with two anchor points.
"""
import argparse
from scipy.ndimage import map_coordinates
import nibabel as nib
import numpy as np
from linumpy.utils.coordinates import AXIS_NAME_TO_INDEX, slice_along_axis
from tqdm import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_nifti',
                   help='Input NIFTI file to be aligned.')
    p.add_argument('out_nifti',
                   help='Output NIFTI file after alignment.')
    p.add_argument('anchor_a', nargs=3, type=float,
                   help='First anchor point (x y z) in voxel coordinates.')
    p.add_argument('anchor_b', nargs=3, type=float,
                   help='Second anchor point (x y z) in voxel coordinates.')
    p.add_argument('--axis', choices=['sagittal', 'coronal', 'axial'],
                   default='sagittal',
                   help='Axis onto which alignment is done.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = nib.load(args.in_nifti)
    p0 = np.asarray(args.anchor_a).reshape((3, 1))
    p1 = np.asarray(args.anchor_b).reshape((3, 1))

    b_hat = np.zeros((3, 1))
    b_hat[AXIS_NAME_TO_INDEX[args.axis]] = 1.0
    a_vec = p1 - p0
    a_hat = a_vec / np.linalg.norm(a_vec)

    cos_theta = a_hat.T.dot(b_hat)
    d_vec = 1.0 / cos_theta * a_hat - b_hat

    axes_2d = [i for i in range(len(in_image.shape)) if i != AXIS_NAME_TO_INDEX[args.axis]]
    shape_2d = [in_image.shape[i] for i in axes_2d]
    # TODO: Handle buffer space better (very memory intensive)
    inds_image = np.stack(np.meshgrid(np.arange(-1.1*shape_2d[0], 1.1*shape_2d[0]),
                                      np.arange(-1.1*shape_2d[1], 1.1*shape_2d[1]), indexing='ij'), axis=0)

    corner_min = np.array([[np.inf, np.inf]])
    corner_max = np.array([[-np.inf, -np.inf]])

    image_data = in_image.get_fdata()
    output_image = []
    for idx in tqdm(range(in_image.shape[AXIS_NAME_TO_INDEX[args.axis]])):
        slicer = slice_along_axis(idx, in_image.shape, AXIS_NAME_TO_INDEX[args.axis])
        delta = idx - p0[AXIS_NAME_TO_INDEX[args.axis]]
        translation = delta * d_vec[axes_2d]
        curr_inds = inds_image + translation.reshape((-1, 1, 1))
        current_image = map_coordinates(image_data[slicer], curr_inds)
        output_image.append(current_image)
        nonzero_indices = np.argwhere(current_image > 1E-4)
        corner_min = np.min(np.vstack((nonzero_indices, corner_min)), axis=0)
        corner_max = np.max(np.vstack((nonzero_indices, corner_max)), axis=0)

    corner_min = corner_min.astype(int)
    corner_max = corner_max.astype(int)
    output_image = np.asarray(output_image)
    output_image = output_image[:, corner_min[0]:corner_max[0], corner_min[1]:corner_max[1]]

     # TODO: Validate this line. Data might not be in the right order if axis is not 0...
    output_image = np.moveaxis(output_image, 0, AXIS_NAME_TO_INDEX[args.axis])

    nib.save(nib.Nifti1Image(output_image, in_image.affine), args.out_nifti)


if __name__ == '__main__':
    main()
