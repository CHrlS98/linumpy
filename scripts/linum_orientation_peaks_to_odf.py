#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Convert principal directions (peaks) to orientation distribution functions.
"""
import argparse
import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from dipy.core.sphere import Sphere


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
                   help='Input peaks nifti image.')
    p.add_argument('in_reference',
                   help='Reference image for estimating hist-FOD. \nOnly non-zero voxels will be evaluated.')
    p.add_argument('out_sh',
                   help='Output spherical harmonics (hist-FOD) image.')

    p.add_argument('--out_normalized',
                   help='Optional normalized SH image (maximum amplitude is 1\n'
                        'for all non-zero voxels).')
    p.add_argument('--weights',
                   help='Optional weights to assign to peak directions.')
    p.add_argument('--exponent', default=1.0, type=float,
                   help='Exponent to apply to weights. [%(default)s]')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Only used if weights is provided. Voxels below the threshold\n'
                        'won\'t be considered for analysis. [%(default)s]')
    p.add_argument('--sh_order_max', type=int, default=6,
                   help='SH order for hist-FOD. [%(default)s]')
    p.add_argument('--sh_basis', choices=SH_BASES.keys(), default='tournier07',
                   help='SH basis for hist-FOD. [%(default)s]')
    return p


def apply_transform(ref_index, in_ref, affine_im_inv, in_im):
    i_ref, j_ref, k_ref = [i - 0.5 for i in ref_index]
    vox_ref_initial = np.array([i_ref, j_ref, k_ref], dtype=float).reshape((3, 1))
    vox_ref_final = np.array([i_ref+1, j_ref+1, k_ref+1], dtype=float).reshape((3, 1))

    world_ref_initial = apply_affine(in_ref.affine, vox_ref_initial.reshape((1, 3)))
    world_ref_final = apply_affine(in_ref.affine, vox_ref_final.reshape((1, 3)))

    vox_im_initial = apply_affine(affine_im_inv, world_ref_initial) + 0.5
    vox_im_final = apply_affine(affine_im_inv, world_ref_final) + 0.5

    vox_im_initial = np.clip(vox_im_initial, 0, np.array(in_im.shape[:3]).reshape((1, 3))).astype(int)
    vox_im_final = np.clip(vox_im_final, 0, np.array(in_im.shape[:3]).reshape((1, 3))).astype(int)

    return vox_im_initial.flatten(), vox_im_final.flatten()


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

    peaks = in_im.get_fdata()

    weights = np.ones(peaks.shape[:3])
    if args.weights is not None:
        weights = nib.load(args.weights).get_fdata() ** args.exponent

    mask = weights > args.threshold
    peaks[~mask] = 0

    affine_im_inv = np.linalg.inv(in_im.affine)

    sf_sphere = get_sphere(name='repulsion200')
    basis_type, legacy = SH_BASES[args.sh_basis]
    B_sphere, _ = sh_to_sf_matrix(sf_sphere, basis_type=basis_type,
                                  sh_order_max=args.sh_order_max,
                                  legacy=legacy, return_inv=True)

    out_sh = np.zeros(in_ref.shape + (B_sphere.shape[0],), dtype=np.float32)
    if args.out_normalized:
        out_sh_normalized = np.zeros_like(out_sh)

    in_ref_mask = in_ref.get_fdata() > args.threshold
    indices = np.nonzero(in_ref_mask)

    for (i_ref, j_ref, k_ref) in zip(*indices):
        vox_im_initial, vox_im_final = apply_transform((i_ref, j_ref, k_ref), in_ref,
                                                       affine_im_inv, in_im)

        # we are inside the image domain so we can use the input
        # volume to estimate hist-FOD in reference space
        current_directions = peaks[vox_im_initial[0]:vox_im_final[0],
                                   vox_im_initial[1]:vox_im_final[1],
                                   vox_im_initial[2]:vox_im_final[2]]
        current_weights = weights[vox_im_initial[0]:vox_im_final[0],
                                  vox_im_initial[1]:vox_im_final[1],
                                  vox_im_initial[2]:vox_im_final[2]]
        current_directions = np.reshape(current_directions, (-1, 3))
        n_elements = len(current_directions)  # size of region for estimating ODF

        # remove null directions and normalize remaining directions
        current_dirnorms = np.linalg.norm(current_directions, axis=-1)

        # if all directions are 0 we skip this voxel
        if not np.any(current_dirnorms > 0):
            continue
        current_directions = current_directions[current_dirnorms > 0]
        current_directions = current_directions / current_dirnorms[current_dirnorms > 0].reshape((-1, 1))

        # create sphere containing directions present at location
        sphere = Sphere(xyz=current_directions)

        current_weights = np.reshape(current_weights, (-1,))
        sf = current_weights[current_dirnorms > 0] / n_elements

        dirac_sh_coeffs = sh_to_sf_matrix(sphere, basis_type=basis_type,
                                          sh_order_max=args.sh_order_max,
                                          legacy=legacy, return_inv=False)
        sh = np.sum(sf.reshape((1, -1))*dirac_sh_coeffs, axis=-1)
        out_sh[i_ref, j_ref, k_ref] = sh.squeeze()

        # optional normalized output for visualization
        if args.out_normalized and np.any(sf > 0):
            sf = np.dot(out_sh[i_ref, j_ref, k_ref], B_sphere)
            out_sh_normalized[i_ref, j_ref, k_ref] = sh.squeeze() / sf.max()

    # save outputs
    nib.save(nib.Nifti1Image(out_sh.astype(np.float32), in_ref.affine), args.out_sh)
    if args.out_normalized:
        nib.save(nib.Nifti1Image(out_sh_normalized.astype(np.float32), in_ref.affine), args.out_normalized)


if __name__ == '__main__':
    main()
