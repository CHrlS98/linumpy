#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reslice nifti volume along arbitrary direction.
"""
import argparse
import nibabel as nib
import numpy as np

from linumpy.utils.coordinates import AXIS_NAME_TO_INDEX
from scipy.interpolate import RegularGridInterpolator

from tqdm import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_nifti',
                   help='Input nifti image to reslice.')
    p.add_argument('out_nifti',
                   help='Resliced nifti image.')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--plane_direction', nargs=3, type=float,
                   metavar=('sx', 'sy', 'sz'),
                   help='Slicing plane direction.')
    g.add_argument('--plane_coordinates', nargs=9, type=float,
                   metavar=('p0x', 'p0y', 'p0z', 'p1x', 'p1y', 'p1z', 'p2x', 'p2y', 'p2z'),
                   help='Coordinates of three points on the slicing plane.')
    group_up_vector = p.add_mutually_exclusive_group()
    group_up_vector.add_argument('--up_vector', nargs=3, type=float, default=(0, 0, 1),
                                 metavar=('ux', 'uy', 'uz'), help='Up vector. [%(default)s]')
    group_up_vector.add_argument('--up_vector_from_coordinates', nargs=9, type=float,
                                 metavar=('p0x', 'p0y', 'p0z', 'p1x', 'p1y', 'p1z', 'p2x', 'p2y', 'p2z'),
                                 help='Coordinates of three points on the plane defined by up_vector.')
    p.add_argument('--axial_resolution', type=float, default=100,
                   help='Axial resolution [%(default)s].')
    p.add_argument('--lateral_resolution', nargs=2, metavar=('rx', 'ry'), type=float, default=(10, 10),
                   help='Output resolution [%(default)s].')
    p.add_argument('--axis', choices=['sagittal', 'coronal', 'axial'],
                   default='sagittal', help='Axis onto which alignment is done.')
    return p


def _get_slicing_vector_world(parser, args, voxel_size):
    if args.plane_direction is not None:
        slicing_vector_world = np.asarray(args.plane_direction) * voxel_size
    elif args.plane_coordinates is not None:
        p0 = np.asarray(args.plane_coordinates[:3])
        p1 = np.asarray(args.plane_coordinates[3:6])
        p2 = np.asarray(args.plane_coordinates[6:])
        p0 *= voxel_size
        p1 *= voxel_size
        p2 *= voxel_size
        slicing_vector_world = np.cross(p1 - p0, p2 - p0)
    else:
        raise parser.error('Either --plane_direction or --plane_coordinates must be provided.')
    slicing_vector_world /= np.linalg.norm(slicing_vector_world)
    return slicing_vector_world


def _get_up_vector_world(args, voxel_size):
    if args.up_vector_from_coordinates is not None:
        p0 = np.asarray(args.up_vector_from_coordinates[:3])
        p1 = np.asarray(args.up_vector_from_coordinates[3:6])
        p2 = np.asarray(args.up_vector_from_coordinates[6:])
        p0 *= voxel_size
        p1 *= voxel_size
        p2 *= voxel_size
        up_vector_world = np.cross(p1 - p0, p2 - p0)
    else:
        up_vector_world = np.asarray(args.up_vector) * voxel_size
    up_vector_world /= np.linalg.norm(up_vector_world)
    return up_vector_world


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = nib.load(args.in_nifti)
    image_data = image.get_fdata()
    voxel_size = image.header.get_zooms()[:3]
    slicing_vector_world = _get_slicing_vector_world(parser, args, voxel_size)
    up_vector = _get_up_vector_world(args, voxel_size)
    up_vector_prime = up_vector - up_vector.dot(slicing_vector_world) * slicing_vector_world
    up_vector_prime /= np.linalg.norm(up_vector_prime)
    leftright_vector = np.cross(up_vector_prime, slicing_vector_world)

    corners = np.array([[0, 0, 0], image.shape]) * voxel_size
    max_extent_world = np.linalg.norm(corners[1] - corners[0])
    print(f'Max extent of the image in world coordinates: {max_extent_world:.2f} um')
    origin = (corners[1] + corners[0]) / 2.0 - (slicing_vector_world + up_vector_prime + leftright_vector) * max_extent_world / 2.0
    leftright_steps, updown_steps = np.meshgrid(
        np.arange(0.0, max_extent_world, args.lateral_resolution[0]),
        np.arange(0.0, max_extent_world, args.lateral_resolution[1]),
        indexing='ij')

    coordinates = origin +\
        leftright_steps[..., None] * leftright_vector +\
        updown_steps[..., None] * up_vector_prime

    grid_interpolator = RegularGridInterpolator(
        (np.arange(image.shape[0])*voxel_size[0],
         np.arange(image.shape[1])*voxel_size[1],
         np.arange(image.shape[2])*voxel_size[2]),
         image_data, method='linear',
         bounds_error=False, fill_value=0)
    
    # we will first stack everything along the first axis
    all_slices = []
    offset = 0.0
    n_steps = int(max_extent_world / args.axial_resolution)
    print(f'Number of slices: {n_steps}')
    for _ in tqdm(range(n_steps)):
        coordinates = coordinates + args.axial_resolution * slicing_vector_world
        offset += args.axial_resolution
        image = grid_interpolator(coordinates)
        all_slices.append(image)

    all_slices = np.asarray(all_slices)
    axis_index = AXIS_NAME_TO_INDEX[args.axis]
    all_slices = np.swapaxes(all_slices, 0, axis_index)

    nonzero_indices = np.argwhere(all_slices > 1E-4)
    corner_min = np.min(nonzero_indices, axis=0).astype(int)
    corner_max = np.max(nonzero_indices, axis=0).astype(int)
    all_slices = all_slices[corner_min[0]:corner_max[0],
                             corner_min[1]:corner_max[1],
                             corner_min[2]:corner_max[2]]

    affine = np.eye(4)
    affine[0, 0] = args.axial_resolution
    affine[1, 1] = args.lateral_resolution[0]
    affine[2, 2] = args.lateral_resolution[1]
    affine[axis_index, axis_index], affine[0, 0] = affine[0, 0], affine[axis_index, axis_index]
    nib.save(nib.Nifti1Image(all_slices, affine), args.out_nifti)

    print(f'Slicing vector (world coordinates) is: {slicing_vector_world}')
    print(f'Up vector prime (world coordinates) is {up_vector_prime}')
    print(f'Left-right vector is {leftright_vector}')
    print(f'Test orthogonality: {up_vector_prime.dot(slicing_vector_world)}')


if __name__ == '__main__':
    main()
