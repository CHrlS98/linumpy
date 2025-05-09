#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute 2D structure tensors for 2D slices from a 3D nifti image.
"""
import argparse

from tqdm import tqdm
from linumpy.orientation.structure_tensor import structure_tensor
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
import nibabel as nib
import numpy as np

from fury import window, actor


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_nifti",
                   help="Path to the input NIfTI file.")
    p.add_argument("in_midplane_landmarks",
                   help="Path to the midplane landmarks NIfTI file.")
    p.add_argument("--principal_direction", default="principal_direction.nii.gz",
                   help="Path to the output principal direction NIfTI file.")
    p.add_argument("--out_rgb", default="rgb.nii.gz",
                   help="DEC map of the structure tensor principal direction [%(default)s].")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="Standard deviation for Gaussian kernel [%(default)s].")
    p.add_argument("--axis", type=int, default=0,
                   help="Axis along which to compute the structure tensor [%(default)s].")
    p.add_argument('--experimental', action='store_true',
                   help='Enable the use of experimental features.')
    return p


def get_mirror_point(pi, p0, n):
    """
    Compute the mirror point of pi with respect to the plane defined by p0 and n.

    Parameters
    ----------
    pi : array_like (..., 3)
        The points to be mirrored.
    p0: array_like (3,)
        A point on the plane.
    n : array_like (3,)
        The normal vector of the plane.
    """
    # Compute the vector from p0 to pi
    v = pi.reshape((-1, 3)) - p0.reshape((1, 3))  # (N, 3)
    # Compute the distance from p0 to the plane
    d = np.dot(v, n.reshape((3, 1)))  # (N, 1)
    # Compute the mirror point
    mirror_point = pi - 2 * d * n.reshape((1, 3))
    return mirror_point.reshape(pi.shape)


def compute_structure_tensor_main_direction(data, axis, sigma):
    _slice = [slice(0, s) for s in data.shape]
    st = np.zeros(data.shape + (2, 2), dtype=np.float32)
    for i in tqdm(range(data.shape[axis])):
        _slice[axis] = i
        curr_slice = data[tuple(_slice)]
        st[tuple(_slice)], (dx, dy) = structure_tensor(curr_slice, sigma=sigma,
                                                       method='gradient',
                                                       gradient_sigma=None)

    # Save the eigenvalues and eigenvectors of the structure tensor
    evals, evecs = np.linalg.eigh(st)
    principal_direction = evecs[..., 0]  # weakest eigenvalue corresponds to the principal direction
    principal_direction_3d = np.zeros(data.shape + (3,), dtype=np.float32)
    principal_direction_3d[..., 1] = principal_direction[..., 0]
    principal_direction_3d[..., 2] = principal_direction[..., 1]

    return principal_direction_3d


def compute_rgb(directions):
    rgb = np.abs(directions)
    rgb -= rgb.min()
    rgb /= rgb.max()
    rgb *= 255
    rgb = rgb.astype(np.uint8)
    return rgb


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = nib.load(args.input_nifti)
    data = image.get_fdata()

    landmark_image = nib.load(args.in_midplane_landmarks)
    landmark_data = landmark_image.get_fdata()

    landmarks_points = np.argwhere(landmark_data > 0)
    landmarks_points = landmarks_points * image.header.get_zooms()

    p0 = landmarks_points[0]
    v0 = landmarks_points[1] - landmarks_points[0]
    v0 = v0 / np.linalg.norm(v0)
    print(f"v0: {v0}")
    v1 = landmarks_points[2] - landmarks_points[0]
    v1 = v1 / np.linalg.norm(v1)
    print(f"v1: {v1}")
    n = np.cross(v0.reshape((1, 3)), v1.reshape((1, 3)))
    n /= np.linalg.norm(n)
    print(f"Normal vector: {n}")

    all_points = np.argwhere(data > 0)
    all_points_um = all_points * image.header.get_zooms()

    all_mirror_points_um = get_mirror_point(all_points_um, p0, n)
    all_mirror_points = all_mirror_points_um / image.header.get_zooms()
    all_mirror_points = np.round(all_mirror_points).astype(np.int32)

    principal_direction_3d = compute_structure_tensor_main_direction(data, args.axis, args.sigma)
    principal_direction_3d[data == 0] = 0

    # flip directions so that they all lie on the same side of the plane
    principal_direction_3d[np.dot(principal_direction_3d, np.array([1.0, 1.0, 0.0])) < 0.0] *= -1

    # visualize
    peaks_actor = actor.peak_slicer(principal_direction_3d[..., None, :]*200,
                                    colors=None, affine=image.affine, linewidth=2.0)
    bg_actor = actor.slicer(data, affine=image.affine, interpolation='nearest', opacity=0.7)
    peaks_actor.display_extent(0, data.shape[0], 0, data.shape[1], data.shape[2]//2, data.shape[2]//2)
    bg_actor.display_extent(0, data.shape[0], 0, data.shape[1], data.shape[2]//2, data.shape[2]//2)
    scene = window.Scene()
    scene.add(peaks_actor, bg_actor)
    window.show(scene, size=(800, 800), reset_camera=False)

    # [EXPERIMENTAL] Flip directions and add them to obtain a 3D direction
    if args.experimental:
        principal_direction_3d = principal_direction_3d * data[..., None]
        mirror_image = np.zeros_like(principal_direction_3d)
        for i in range(3):
            mirror_image[all_points[:, 0], all_points[:, 1], all_points[:, 2], i] =\
                map_coordinates(principal_direction_3d[..., i],
                                all_mirror_points.T, order=1,
                                mode='nearest')
        mirror_image = mirror_image - 2.0 * mirror_image.dot(n.reshape((3, 1))) * n.reshape((1, 1, 1, 3))

        principal_direction_3d = 0.5 * principal_direction_3d + 0.5 * mirror_image

    # Save the RGB image
    rgb = compute_rgb(principal_direction_3d)
    nib.save(nib.Nifti1Image(rgb, image.affine), args.out_rgb.replace(".nii.gz", "_rgb_native.nii.gz"))

    # rotation to align the plane with the coronal plane (standard DEC map)
    rotation = Rotation.from_rotvec(np.array([0.0, 0.0, 1.0])*-np.pi/4).as_matrix()
    principal_direction_3d = principal_direction_3d.dot(rotation.T)
    # convert to DEC map
    rgb = compute_rgb(principal_direction_3d)
    nib.save(nib.Nifti1Image(rgb, image.affine), args.out_rgb.replace(".nii.gz", "_rgb_dec.nii.gz"))


if __name__ == "__main__":
    main()
