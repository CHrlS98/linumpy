#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Perform structure tensor analysis on Nifti image.
"""
import argparse
from skimage.feature import structure_tensor
import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import itertools
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere


SH_BASES = {
    'descoteaux07_legacy': ('descoteaux07', True),
    'tournier07_legacy': ('tournier07', True),
    'descoteaux07': ('descoteaux07', False),
    'tournier07': ('tournier07', False)
}


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input nifti image.')
    p.add_argument('in_reference',
                   help='Reference image for estimating hist-FOD.')
    p.add_argument('out_evals',
                   help='Output eigenvalues nifti image.')
    p.add_argument('out_evecs',
                   help='Output eigenvectors nifti image.')
    p.add_argument('out_sh',
                   help='Output spherical harmonics (hist-FOD) image.')
    p.add_argument('--split_evecs_prefix',
                   help='If supplied, eigenvectors are also saved separately with this prefix.')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Voxels below the threshold won\'t be considered for analysis. [%(default)s]')
    p.add_argument('--sigma', type=float, default=1.0,
                   help='Standard deviation of Gaussian windowing function. [%(default)s]')
    p.add_argument('--sh_order_max', type=int, default=6,
                   help='SH order for hist-FOD. [%(default)s]')
    p.add_argument('--sh_basis', choices=SH_BASES.keys(), default='tournier07',
                   help='SH basis for hist-FOD. [%(default)s]')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_im = nib.load(args.in_image)
    in_ref = nib.load(args.in_reference)

    # Test header compatibility
    mat_im = in_im.affine[:3, :3]
    mat_ref = in_ref.affine[:3, :3]
    scale_im_inv = np.diag(1.0 / np.asarray(in_im.header.get_zooms()[:3]))
    scale_ref_inv = np.diag(1.0 / np.asarray(in_ref.header.get_zooms()[:3]))

    rot_im = mat_im @ scale_im_inv
    rot_ref = mat_ref @ scale_ref_inv
    if not np.allclose(rot_im, rot_ref):
        print('WARNING: Script assumes equivalent rotation between Image and Reference but test failed.\n'
              '         Make sure Image and Reference have equivalent rotation.')

    in_data = in_im.get_fdata()

    A_00, A_01, A_02, A_11, A_12, A_22 = structure_tensor(in_data, args.sigma)

    A = np.zeros(in_data.shape + (3, 3))
    A[..., 0, 0] = A_00
    A[..., 0, 1] = A[..., 1, 0] = A_01
    A[..., 0, 2] = A[..., 2, 0] = A_02
    A[..., 1, 1] = A_11
    A[..., 1, 2] = A[..., 2, 1] = A_12
    A[..., 2, 2] = A_22

    mask = in_data > args.threshold

    evals, evecs = np.linalg.eigh(A[mask])

    evals_full = np.zeros(in_data.shape + (3,))
    evals_full[mask] = evals

    evecs_full = np.zeros(in_data.shape + (3, 3))
    evecs_full[mask] = evecs

    pdir = evecs_full[..., 0]

    affine_im_inv = np.linalg.inv(in_im.affine)
    print('here')

    sf_sphere = get_sphere(name='repulsion200')
    basis_type, legacy = SH_BASES[args.sh_basis]
    _, B_inv = sh_to_sf_matrix(sf_sphere, basis_type=basis_type, sh_order_max=args.sh_order_max,
                                   legacy=legacy, return_inv=True, smooth=0.0001)

    out_sh = np.zeros(in_ref.shape + (B_inv.shape[1],), dtype=np.float32)

    for (i_ref, j_ref, k_ref) in itertools.product(*[range(in_ref.shape[i]) for i in range(len(in_ref.shape))]):
        vox_ref_initial = np.array([i_ref, j_ref, k_ref], dtype=float).reshape((3, 1))
        vox_ref_final = np.array([i_ref+1, j_ref+1, k_ref+1], dtype=float).reshape((3, 1))

        world_ref_initial = apply_affine(in_ref.affine, vox_ref_initial.reshape((1, 3)))
        world_ref_final = apply_affine(in_ref.affine, vox_ref_final.reshape((1, 3)))

        vox_im_initial = apply_affine(affine_im_inv, world_ref_initial)
        vox_im_final = apply_affine(affine_im_inv, world_ref_final)

        # nearest neighbour
        vox_im_initial = np.floor(vox_im_initial).astype(int).squeeze()
        vox_im_final = np.floor(vox_im_final).astype(int).squeeze()

        # test if we are inside the image
        if np.any(vox_im_initial < 0) or np.any(vox_im_final) < 0:
            continue
        if np.any(vox_im_initial > np.reshape(in_im.shape, vox_im_initial.shape)) or\
            np.any(vox_im_final > np.reshape(in_im.shape, vox_im_final.shape)):
            continue

        # we are inside the image domain so we can use the input
        # volume to estimate hist-FOD in reference space
        directions = pdir[vox_im_initial[0]:vox_im_final[0],
                          vox_im_initial[1]:vox_im_final[1],
                          vox_im_initial[2]:vox_im_final[2]]
        weights = in_data[vox_im_initial[0]:vox_im_final[0],
                          vox_im_initial[1]:vox_im_final[1],
                          vox_im_initial[2]:vox_im_final[2]]
        directions = np.reshape(directions, (-1, 3))
        weights = np.reshape(weights, (-1,))

        directions_to_vertices = np.abs(directions.dot(sf_sphere.vertices.T))
        bins = np.argmax(directions_to_vertices, axis=1)
        sf = np.zeros(len(sf_sphere.vertices))
        sf[bins] = weights
        sh = sf.reshape((1, -1)).dot(B_inv)
        out_sh[i_ref, j_ref, k_ref] = sh.squeeze()

    # save outputs
    nib.save(nib.Nifti1Image(out_sh.astype(np.float32), in_ref.affine), args.out_sh)
    nib.save(nib.Nifti1Image(evals_full.astype(np.float32), in_im.affine),  args.out_evals)
    nib.save(nib.Nifti1Image(evecs_full.astype(np.float32), in_im.affine), args.out_evecs)

    if args.split_evecs_prefix is not None:
        for i in range(3):
            evec_i = evecs_full[..., i]
            nib.save(nib.Nifti1Image(evec_i.astype(np.float32), in_im.affine), args.split_evecs_prefix + f'{i}.nii.gz')


if __name__ == '__main__':
    main()
