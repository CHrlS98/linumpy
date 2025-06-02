#!/usr/bin/env python3
import argparse
from fury import window, actor
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
import numpy as np
import nibabel as nib


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Input SH nifti.')
    p.add_argument('in_oct',
                   help='Input OCT image.')
    p.add_argument('out_image',
                   help='Output image in .png format.')
    p.add_argument('--index', type=int,
                   help='Index of the OCT slice to visualize [%(default)s].')
    p.add_argument('--axis', choices=('sagittal', 'coronal', 'axial'),
                   default='axial',
                   help='Axis of the slice to visualize [%(default)s].')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sh_im = nib.load(args.in_sh)
    oct_im = nib.load(args.in_oct)

    sh_data = sh_im.get_fdata()
    print(sh_data.shape)
    oct_data = oct_im.get_fdata()

    sphere = get_sphere(name='repulsion724').subdivide(n=1)
    B_mat = sh_to_sf_matrix(sphere, sh_order_max=8, return_inv=False)
    print(B_mat.shape)

    if args.axis == 'sagittal':
        axis_to_index = 0
    elif args.axis == 'coronal':
        axis_to_index = 1
    elif args.axis == 'axial':
        axis_to_index = 2
    index = args.index or oct_data.shape[axis_to_index] // 2
    mask = np.sum(np.abs(sh_data), axis=-1) > 1E-8

    x1 = index if args.axis == 'sagittal' else 0
    y1 = index if args.axis == 'coronal' else 0
    z1 = index if args.axis == 'axial' else 0
    x2 = index if args.axis == 'sagittal' else oct_data.shape[0] - 1
    y2 = index if args.axis == 'coronal' else oct_data.shape[1] - 1
    z2 = index if args.axis == 'axial' else oct_data.shape[2] - 1
    oct_actor = actor.slicer(oct_data, interpolation='nearest', affine=oct_im.affine)
    oct_actor.display_extent(x1, x2, y1, y2, z1, z2)
    affine = sh_im.affine
    scaling = oct_im.header.get_zooms()[0] / sh_im.header.get_zooms()[0]
    affine[axis_to_index, -1] += 1.0 * sh_im.header.get_zooms()[0]
    odf_actor = actor.odf_slicer(sh_data, sphere=sphere,
                                 affine=sh_im.affine, mask=mask,
                                 B_matrix=B_mat, norm=True,
                                 scale=0.5)
    odf_actor.display_extent(int(scaling*x1), int(scaling*x2),
                             int(scaling*y1), int(scaling*y2),
                             int(scaling*z1), int(scaling*z2))

    scene = window.Scene()
    scene.add(oct_actor, odf_actor)
    scene.reset_camera_tight()
    scene.camera().ParallelProjectionOn()
    window.show(scene, reset_camera=False)

    window.record(scene=scene, out_path=args.out_image,
                  size=(4000, 3800), reset_camera=False)


if __name__ == '__main__':
    main()
