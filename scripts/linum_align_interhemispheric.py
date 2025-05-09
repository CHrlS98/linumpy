#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given a landmarks mask where values of True correspond to coordinates located
along the brain midplane, fits a polynomial to the coordinates and rotate them such
that the midplane makes a perfect 45 degrees between sagittal and coronal.

TODO: This angle should be configurable to increase generalizability.
"""
import argparse
import nibabel as nib
import numpy as np
from skimage.measure import label
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

# TODO: Add a landmark for anterior position


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_nifti",
                   help="Path to the input NIfTI file.")
    p.add_argument("in_landmarks",
                   help="Path to the input landmarks file.")
    p.add_argument("out_nifti",
                   help="Path to the output NIfTI file.")
    p.add_argument('--show_figure', action='store_true',
                   help='Show a figure showing the plane onto which the'
                        ' slices are aligned.')
    return p


def _polynomial(x1, x2, x1_order, x2_order):
    """
    Compute polynomial features of the input data.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Input x coordinates.
    y : array-like, shape (n_samples,)
        Input y coordinates.
    x1_order : int
        The order of the polynomial features along x axis.
    x2_order : int
        The order of the polynomial features along y axis.
    """
    X_poly = np.ones((1, len(x1)))  # bias term
    for i in range(1, x1_order + 1):
        X_poly = np.vstack((X_poly, x1 ** i))
    for i in range(1, x2_order + 1):
        X_poly = np.vstack((X_poly, x2 ** i))
        for j in range(1, x1_order + 1):
            X_poly = np.vstack((X_poly, x1 ** j * x2 ** i))
    return X_poly


def _evaluate_polynomial(x1, x2, w, x1_order, x2_order):
    psi_X = _polynomial(x1, x2, x1_order, x2_order)
    return np.dot(w.T, psi_X)


def _landmarks_to_positions(landmarks, voxel_size):
    landmarks_segment = np.zeros(landmarks.shape, dtype=int)
    num_labels = 0
    for i in range(landmarks.shape[0]):
        temp_segments, _num_labels = label(landmarks[i], return_num=True)
        temp_segments[temp_segments > 0] += num_labels
        landmarks_segment[i] = temp_segments
        num_labels += _num_labels

    pos_vox = []
    pos_voxmm = []
    for i in range(1, num_labels):
        coords =  np.argwhere(landmarks_segment == i)
        coords = np.mean(coords, axis=0)
        pos_vox.append(coords)
        coords = coords * voxel_size
        pos_voxmm.append(coords)

    pos_voxmm = np.array(pos_voxmm)
    pos_vox = np.array(pos_vox)
    return pos_voxmm, pos_vox


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the input NIfTI file
    in_nifti = nib.load(args.in_nifti)
    voxel_size = in_nifti.header.get_zooms()

    # Load the input landmarks file
    landmarks = nib.load(args.in_landmarks).get_fdata().astype(bool)
    pos_voxmm, pos_vox = _landmarks_to_positions(landmarks, voxel_size)

    # regression parameters
    lamb = 0.0
    x1_order = 2
    x2_order = 1

    Y = np.reshape(pos_voxmm[:, 1], (1, -1))  # \in R^{1 x n}
    psi_X = _polynomial(pos_voxmm[:, 0], pos_voxmm[:, 2], x1_order, x2_order)
    w = np.linalg.inv(psi_X @ psi_X.T + lamb * np.eye(psi_X.shape[0])) @ psi_X @ Y.T

    min_pos = np.min(pos_voxmm, axis=0)
    max_pos = np.max(pos_voxmm, axis=0)

    x, y = np.meshgrid(np.arange(min_pos[0], max_pos[0], 500),
                       np.arange(min_pos[2], max_pos[2], 500),
                       indexing='ij')
    Y_pred = _evaluate_polynomial(x.flatten(), y.flatten(), w, x1_order, x2_order)

    if args.show_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x.flatten(), Y_pred.flatten(), y.flatten(), color='blue')
        ax.scatter(x.flatten(), x.flatten(), y.flatten(), color='green')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        return

    input_volume = in_nifti.get_fdata()
    output_volume = []

    corner_min = np.array([[np.inf, np.inf]])
    corner_max = np.array([[-np.inf, -np.inf]])
    for i_slice in np.arange(landmarks.shape[0]):
        slice_offset = i_slice * voxel_size[0]

        z0 = min_pos[2]
        z1 = max_pos[2]
        y_pred = _evaluate_polynomial(np.array([slice_offset, slice_offset]),
                                      np.array([z0, z1]), w, x1_order, x2_order)
        y0 = y_pred[0, 0]
        y1 = y_pred[0, 1]
        slope = (y1 - y0) / (z1 - z0)
        bias = y1 - slope*z1
        z_star = (slice_offset - bias) / slope
        anchor = np.array([slice_offset / voxel_size[1], z_star / voxel_size[2]])

        # Add buffer coordinates so we are sure to cover the whole slice.
        # TODO: FIX ME. This is too memory intensive.
        coordinates = np.stack((np.meshgrid(np.arange(-2*landmarks.shape[1], 2*landmarks.shape[1]),
                                            np.arange(-2*landmarks.shape[2], 2*landmarks.shape[2]),
                                            indexing='ij')), axis=0)
        coords_shape = coordinates.shape

        angle = -np.arctan2(y1 - y0, z1 - z0)
        rotmat = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
        coordinates = coordinates - anchor[:, None, None]
        coordinates = np.reshape(rotmat @ coordinates.reshape(2, -1), coords_shape)
        coordinates = coordinates + anchor[:, None, None]
        out_slice = map_coordinates(input_volume[i_slice], coordinates)
        nonzero_indices = np.argwhere(out_slice > 1E-4)
        corner_min = np.min(np.vstack((nonzero_indices, corner_min)), axis=0)
        corner_max = np.max(np.vstack((nonzero_indices, corner_max)), axis=0)

        output_volume.append(map_coordinates(input_volume[i_slice], coordinates))

    corner_min = corner_min.astype(int)
    corner_max = corner_max.astype(int)
    output_volume = np.asarray(output_volume)
    output_volume = output_volume[:, corner_min[0]:corner_max[0], corner_min[1]:corner_max[1]]

    nib.save(nib.Nifti1Image(output_volume, in_nifti.affine), args.out_nifti)


if __name__ == "__main__":
    main()