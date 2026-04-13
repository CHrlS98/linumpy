#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Convert principal directions (peaks) to orientation distribution functions.
"""
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter
from dipy.reconst.shm import sh_to_sf_matrix, sph_harm_ind_list, sh_to_rh
from dipy.core.sphere import hemi_icosahedron
from linumpy.utils.io import assert_output_exists, add_overwrite_arg


SH_BASES = {
    'descoteaux07_legacy': ('descoteaux07', True),
    'tournier07_legacy': ('tournier07', True),
    'descoteaux07': ('descoteaux07', False),
    'tournier07': ('tournier07', False)
}


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_sh',
                   help='Output spherical harmonics (hist-FOD) image.')
    p.add_argument('--in_peaks', nargs='+', required=True,
                   help='Input peaks nifti image.')
    p.add_argument('--in_certainty', nargs='+',
                   help='Certainty image used to assign weight to directions.')

    p.add_argument('--brain_mask',
                   help='Optional nifti image to mask the output hist-FOD. Only non-zero voxels will be evaluated.')
    p.add_argument('--sh_order_max', type=int, default=8,
                   help='SH order for hist-FOD. [%(default)s]')
    p.add_argument('--sh_basis', choices=SH_BASES.keys(), default='tournier07',
                   help='SH basis for hist-FOD. [%(default)s]')
    p.add_argument('--apodized_delta', action='store_true',
                   help='Use apodized delta kernel for mapping peaks to SH coefficients. [%(default)s]')
    p.add_argument('--normalize_certainty', action='store_true',
                   help='Normalize SH amplitudes by sum of certainties to avoid bias\n'
                        'towards voxels with more certainty. [%(default)s]')

    p.add_argument('--width', type=int,
                   help='Width for average/maximum smoothing.')
    p.add_argument('--maximum_filter', action='store_true',
                   help='Use maximum filter instead of uniform filter for smoothing. [%(default)s]')
    p.add_argument('--padding_mode', choices=['constant', 'reflect', 'nearest'], default='constant',
                   help='Mode for padding. [%(default)s]')
    p.add_argument('--padding_cval', type=float, default=0.0,
                   help='Value for padding. [%(default)s]')
    add_overwrite_arg(p)
    return p


def generate_apodized_delta(sh_order_max, basis_type, legacy, reg=1000, f=0.1):
    sphere = hemi_icosahedron.subdivide(n=5)
    print(f'Number of vertices in sphere: {len(sphere.vertices)}')
    Q = sh_to_sf_matrix(sphere, sh_order_max=sh_order_max,
                        basis_type=basis_type, legacy=legacy,
                        return_inv=False).T
    mask_theta_0 = np.abs(sphere.vertices.dot([0, 0, 1])) == 1

    print('Number of theta = 0:', np.count_nonzero(mask_theta_0))

    p_n = Q[mask_theta_0].reshape((-1, 1))
    p_np1 = None
    z = mask_theta_0.astype(float).reshape((-1, 1))
    n = 0
    while n < 100:  # Add a maximum iteration limit
        a_n = Q.dot(p_n)
        neg_mask = a_n < f * p_n[0, 0] / np.sqrt(4.0*np.pi)
        if not np.any(neg_mask):
            print('No values below threshold, stopping iteration.')
            break
        L_n = np.diag(neg_mask.flatten().astype(float))
        pinv = np.linalg.pinv((np.eye(L_n.shape[0]) + reg * L_n).dot(Q))
        p_np1 = pinv.dot(z)
        p_n = p_np1.copy()
        n = n + 1
    p_n = p_n / (p_n[0, 0] * np.sqrt(4.0*np.pi))
    print(f'Converged in {n} iterations.')

    m_list, l_list = sph_harm_ind_list(sh_order_max)
    kern = np.zeros(int((sh_order_max+1)*(sh_order_max+2)//2), dtype=np.float32)
    rh = sh_to_rh(p_n.flatten(), m_list, l_list)
    for i, l in enumerate(np.arange(sh_order_max+1, step=2)):
        kern[l_list == l] = rh[i]
    return kern  # return as float32 to reduce memory usage


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    basis_type, legacy = SH_BASES[args.sh_basis]

    assert_output_exists(args.out_sh, parser, args)

    brain_mask = None
    if args.brain_mask is not None:
        mask_im = nib.load(args.brain_mask)
        brain_mask = mask_im.get_fdata().astype(bool)

    apodize_kernel = None
    if args.apodized_delta:
        apodize_kernel = generate_apodized_delta(args.sh_order_max, basis_type, legacy)

    sphere = hemi_icosahedron.subdivide(n=4)
    sh_to_sf_mat = sh_to_sf_matrix(sphere, sh_order_max=args.sh_order_max,
                                   basis_type=basis_type, legacy=legacy,
                                   return_inv=False).astype(np.float32)

    ref_peaks_im = nib.load(args.in_peaks[0])
    peaks = ref_peaks_im.get_fdata().astype(np.float32)  # force float32 to save memory
    certainty = np.ones(peaks.shape[:-1], dtype=np.float32)

    # numpy.zeros does not actually allocate memory until values are assigned.
    out_sh = np.zeros(ref_peaks_im.shape[:-1] + (sh_to_sf_mat.shape[0],), dtype=np.float32)
    out_sh[:] = 0  # force allocation of memory

    certainty_sums = np.zeros(ref_peaks_im.shape[:-1], dtype=np.float32)

    for peak_id in range(len(args.in_peaks)):
        print(f"Processing peak {peak_id+1}/{len(args.in_peaks)}")
        print('Peaks image:', args.in_peaks[peak_id])
        print('Certainty image:', args.in_certainty[peak_id] if args.in_certainty is not None else 'None')
        peaks = nib.load(args.in_peaks[peak_id]).get_fdata()
        if args.in_certainty is not None:
            in_certainty_im = nib.load(args.in_certainty[peak_id])
            if in_certainty_im.shape != peaks.shape[:-1]:
                parser.error(f'{args.in_certainty} shape mismatch. All images must have the same 3D shape!!')
            certainty = in_certainty_im.get_fdata()

        # process peak
        peaks_norm = np.linalg.norm(peaks, axis=-1)
        mask = (peaks_norm > 0) & (certainty > 0)
        if brain_mask is not None:
            mask = mask & brain_mask
        peaks1d = peaks[mask]
        certainty1d = certainty[mask]
        certainty1d = certainty1d / np.max(certainty1d)  # normalize certainty to [0, 1]

        peaks1d_to_sph_ind = np.zeros((peaks1d.shape[0],), dtype=int)
        max_dot = np.zeros((peaks1d.shape[0],), dtype=float)
        for vert_idx, d in enumerate(sphere.vertices):
            if vert_idx % 50 == 0:
                print(f"Processing sphere vertex {vert_idx}/{len(sphere.vertices)}")
            dot = np.abs(peaks1d.dot(d))
            update = dot > max_dot
            max_dot[update] = dot[update]
            peaks1d_to_sph_ind[update] = vert_idx

        # THIS LINE IS VERY MEMORY INTENSIVE: OOM ON LARGE IMAGES
        sh = sh_to_sf_mat.T[peaks1d_to_sph_ind]
        if apodize_kernel is not None:
            sh = sh * apodize_kernel

        out_sh[mask] += sh * certainty1d[:, None]
        certainty_sums[mask] += certainty1d

    # Normalize SH amplitudes by sum of certainties to
    # avoid bias towards voxels with more certainty
    if args.normalize_certainty:
        certainty_sums[certainty_sums == 0] = 1  # to avoid division by zero
        out_sh = out_sh / certainty_sums[..., None]

    if args.width is not None and args.width > 1:
        print("Post-filtering of SH coefficients...")
        # Use maximum filter to avoid cancelling out peaks in the average
        if args.maximum_filter:
            sphere = hemi_icosahedron.subdivide(n=2)
            sh_to_sf_mat, sf_to_sh_mat = sh_to_sf_matrix(sphere, sh_order_max=args.sh_order_max,
                                                         basis_type=basis_type, legacy=legacy)
            sf = np.dot(out_sh, sh_to_sf_mat)
            sf = maximum_filter(sf, args.width, axes=(0, 1, 2),
                                mode=args.padding_mode, cval=args.padding_cval)
            out_sh = np.dot(sf, sf_to_sh_mat)
        else:
            out_sh = uniform_filter(out_sh, size=args.width, mode=args.padding_mode,
                                    cval=args.padding_cval, axes=(0, 1, 2))

    # scale by voxel size for compatibility with MRtrix
    vox_res = np.mean(ref_peaks_im.header.get_zooms()[:3])
    out_sh = out_sh * vox_res

    # Mask output if brain mask provided
    if brain_mask is not None:
        out_sh[~brain_mask] = 0

    print("Saving output...")
    nib.save(nib.Nifti1Image(out_sh.astype(np.float32), ref_peaks_im.affine),
             args.out_sh)


if __name__ == '__main__':
    main()
