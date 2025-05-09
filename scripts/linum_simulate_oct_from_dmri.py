"""
This script simulates Optical Coherence Tomography (OCT) data from Diffusion MRI (dMRI) data.
"""
import numpy as np
import argparse
import nibabel as nib
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere, HemiSphere
from tqdm import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_fodf',
                   help='Path to the input FODF image.')
    p.add_argument('out_oct',
                   help='Path to the output OCT image.')
    p.add_argument('--fa',
                   help='Path to the optional FA image.')
    p.add_argument('--slicing_direction', nargs=3, type=float,
                    default=[1.0, 1.0, 0.0],
                    help='Direction of the slicing plane [%(default)s].')
    p.add_argument('--dest_axis', choices=['x+', 'y+', 'z+', 'x-', 'y-', 'z-'], default='x+',
                    help='Axis along which the slices are stacked.')
    p.add_argument('--bg_threshold', type=float, default=0.0,
                    help='Background threshold for FODF [%(default)s].')
    p.add_argument('--sharpening', type=float, default=1.0,
                    help='Sharpening factor for the OCT image [%(default)s].')
    p.add_argument('--reslice', action='store_true',
                    help='Reslice the OCT image along the slicing direction.')
    return p


def generate_classical(nonzero_fodf, nonzero_weights, b_matrix,
                       sphere, slicing_direction, batch_size=8000):
    intensities = np.zeros((len(nonzero_fodf)))
    # (N, 1) shape
    dot_product = np.dot(sphere.vertices.reshape((-1, 3)), slicing_direction)
    sin_theta = np.sqrt(1 - dot_product ** 2)
    n_batches = int(np.ceil(len(nonzero_fodf) / batch_size))
    for batch in tqdm(range(n_batches)):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(nonzero_fodf))
        sh_i_batch = nonzero_fodf[start:end]
        # (N, 1) shape
        sf_i_batch = sh_i_batch @ b_matrix
        # (N, 1) shape
        intensities[start:end] =\
            np.sum(sf_i_batch * sin_theta.reshape((1, -1)), axis=1) *\
            nonzero_weights[start:end]

    return intensities


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    fodf_img = nib.load(args.in_fodf)
    fodf = fodf_img.get_fdata()

    fa = np.zeros(fodf.shape[:-1])
    if args.fa is not None:
        fa = nib.load(args.in_fa).get_fdata()
    weights = (fa + 1.0)**args.sharpening

    # 1. convert sh to sf using dipy
    sphere = HemiSphere.from_sphere(get_sphere(name='symmetric724'))

    slicing_direction = np.array(args.slicing_direction, dtype=np.float32).reshape(3, 1)
    slicing_direction /= np.linalg.norm(slicing_direction)

    B_matrix, _ = sh_to_sf_matrix(sphere, sh_order_max=8)

    # 2. Simulate signal from FODF
    nonzero_fodf = fodf[fodf[..., 0] > args.bg_threshold]
    nonzero_weights = weights[fodf[..., 0] > args.bg_threshold]
    intensities = generate_classical(nonzero_fodf, nonzero_weights, B_matrix,
                                     sphere, slicing_direction)

    oct = np.zeros(fodf_img.shape[:-1])
    oct[fodf[..., 0] > args.bg_threshold] = intensities

    if args.reslice:
        # Reslice image along the slicing direction
        center_coord = np.array(oct.shape) / 2.0
        radius = np.max(oct.shape) / 2.0
        coordinates = np.stack(np.meshgrid(
            np.arange(center_coord[0]-radius, center_coord[0]+radius),
            np.arange(center_coord[1]-radius, center_coord[1]+radius),
            np.arange(center_coord[2]-radius, center_coord[2]+radius),
            indexing='ij'), axis=0).astype(np.float32)
        dest_axis = np.zeros((3, 1), dtype=np.float32)
        if 'x' in args.dest_axis:
            dest_axis[0] = 1.0
        elif 'y' in args.dest_axis:
            dest_axis[1] = 1.0
        elif 'z' in args.dest_axis:
            dest_axis[2] = 1.0
        if '-' in args.dest_axis:
            dest_axis *= -1.0

        rotvec = np.cross(slicing_direction.flatten(), dest_axis.flatten())
        if np.linalg.norm(rotvec) > 0.0:
            coordinates -= center_coord[:, None, None, None]
            rotvec /= np.linalg.norm(rotvec)
            theta = np.arccos(np.dot(slicing_direction.flatten(), dest_axis.flatten()))
            _rotmat  = R.from_rotvec(rotvec * theta).as_matrix()
            rotmat = np.eye(4)
            rotmat[:3, :3] = _rotmat
            coordinates = np.concatenate((coordinates, np.ones((1, *coordinates.shape[1:]))), axis=0)
            coordinates = np.tensordot(rotmat, coordinates, axes=(0, 0))
            coordinates = coordinates[:3] + center_coord[:, None, None, None]

        oct = map_coordinates(oct, coordinates)
        mask = oct > 1E-8
        oct[~mask] = 0
        valid_voxels = np.argwhere(mask)
        min_corner = np.min(valid_voxels, axis=0)
        max_corner = np.max(valid_voxels, axis=0)
        oct = oct[min_corner[0]:max_corner[0]+1,
                  min_corner[1]:max_corner[1]+1,
                  min_corner[2]:max_corner[2]+1]

    affine = np.eye(4)
    affine[0, 0] = fodf_img.header.get_zooms()[0] * 1000  # in um
    affine[1, 1] = fodf_img.header.get_zooms()[1] * 1000  # in um
    affine[2, 2] = fodf_img.header.get_zooms()[2] * 1000  # in um
    oct_img = nib.Nifti1Image(oct, affine)
    nib.save(oct_img, args.out_oct)
    print(f"Saved simulated OCT image to {args.out_oct}")


if __name__ == "__main__":
    main()