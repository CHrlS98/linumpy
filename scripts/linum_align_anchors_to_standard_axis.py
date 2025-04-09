#!/usr/bin/env python3
import argparse
from scipy.ndimage import map_coordinates
import nibabel as nib
import numpy as np
from linumpy.utils.coordinates import AXIS_NAME_TO_INDEX, slice_along_axis


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_nifti')
    p.add_argument('out_nifti')
    p.add_argument('anchor_a', nargs=3, type=float)
    p.add_argument('anchor_b', nargs=3, type=float)
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
    inds_image = np.stack(np.meshgrid(np.arange(shape_2d[0]), np.arange(shape_2d[1]), indexing='ij'), axis=0)

    image_data = in_image.get_fdata()
    out_image = np.zeros_like(image_data)
    for idx in range(in_image.shape[AXIS_NAME_TO_INDEX[args.axis]]):
        slicer = slice_along_axis(idx, in_image.shape, AXIS_NAME_TO_INDEX[args.axis])
        delta = idx - p0[AXIS_NAME_TO_INDEX[args.axis]]
        translation = delta * d_vec[axes_2d]
        curr_inds = inds_image + translation.reshape((-1, 1, 1))
        out_image[slicer] = map_coordinates(image_data[slicer], curr_inds)

    nib.save(nib.Nifti1Image(out_image, in_image.affine), args.out_nifti)


if __name__ == '__main__':
    main()
