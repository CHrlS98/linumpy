#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the FOD estimation method described in [1] based on
structure tensor analysis.
"""
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.direction.peaks import reshape_peaks_for_visualization

from linumpy.preproc.icorr import normalize
from linumpy.io.zarr import read_omezarr
from linumpy.feature.orientation import\
    _make_xfilter, _make_yfilter, _make_zfilter


EPILOG="""
[1] Schilling et al, "Comparison of 3D orientation distribution functions
    measured with confocal microscopy and diffusion MRI". Neuroimage. 2016
    April 1; 129: 185-197. doi:10.1016/j.neuroimage.2016.01.022
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input image in .ome.zarr file format.')
    p.add_argument('out_sh',
                   help='Output SH image (.nii.gz).')
    p.add_argument('--sigma', default=0.010, type=float,
                   help='Standard deviation of derivative of Gaussian (mm).')
    p.add_argument('--rho', default=0.025, type=float,
                   help='Standard deviation of Gaussian window for '
                        'structure tensor estimation (mm).')
    p.add_argument('--new_voxel_size', default=0.1, type=float,
                   help='Size of voxels for histological-FOD (mm).')
    p.add_argument('--damp', default=1.0, type=float,
                   help='Dampening factor for weighting structure tensor directions '
                        'based on the certainty measure `c_p`. Each tensor direction'
                        ' will be weighted by `c_p**damp` prior to binning.')
    return p


def samples_from_sigma(sigma):
    return np.arange(-int(np.ceil(sigma * 3)), int(np.ceil(sigma * 3)) + 1)


def gaussian(sigma):
    r = samples_from_sigma(sigma)
    ret = 1.0 / np.sqrt(2.0 * np.pi * sigma**2) * np.exp(-r**2 / 2.0 / sigma**2)
    ret = 1.0 / np.sqrt(np.sum(ret**2)) * ret
    print(np.sum(ret**2))
    return ret


def gaussian_derivative(sigma):
    r = samples_from_sigma(sigma)
    ret = -r / sigma**2 * gaussian(sigma)
    ret = 1.0 / np.sqrt(np.sum(ret**2)) * ret
    print(np.sum(ret**2))
    return ret


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    data, res = read_omezarr(args.in_image, level=0)
    data = np.moveaxis(data, (0, 1, 2), (2, 1, 0))
    data = normalize(data, 1, 99.99)

    sigma_to_vox = np.array([args.sigma]) / np.asarray(res)[::-1]
    rho_to_vox = np.array([args.rho]) / np.asarray(res)[::-1]
    new_voxel_size_to_vox = (np.array([args.new_voxel_size]) / np.asarray(res)[::-1]).astype(int)
    n_chunks_per_axis = np.ceil(np.asarray(data.shape) / np.asarray(new_voxel_size_to_vox)).astype(int)

    # 1. Estimate derivatives f_x, f_y, f_z
    # For estimating derivatives
    dx = _make_xfilter(gaussian_derivative(sigma_to_vox[0]))
    dy = _make_yfilter(gaussian_derivative(sigma_to_vox[1]))
    dz = _make_zfilter(gaussian_derivative(sigma_to_vox[2]))
    print(dx)

    # Windowing function
    gaussian_filters = []
    gaussian_filters.append(_make_xfilter(gaussian(rho_to_vox[0])))
    gaussian_filters.append(_make_yfilter(gaussian(rho_to_vox[1])))
    gaussian_filters.append(_make_zfilter(gaussian(rho_to_vox[2])))

    derivatives = []
    derivatives.append(convolve(data, dx, mode='wrap'))
    derivatives.append(convolve(data, dy, mode='wrap'))
    derivatives.append(convolve(data, dz, mode='wrap'))

    # 2. Build structure tensor
    ST = np.zeros(data.shape + (3, 3))
    for i in range(3):
        for j in np.arange(i, 3):
            derivative = derivatives[i] * derivatives[j]
            for g_filter in gaussian_filters:
                derivative = convolve(derivative, g_filter)
            ST[..., i, j] = derivative
            ST[..., j, i] = derivative

    evals, evecs = np.linalg.eigh(ST)
    peaks = np.swapaxes(evecs, -2, -1)
    p = peaks[..., 0, :][..., None, :]

    # at the difference of Schilling et al (2016) here we use
    # the certainty measure to weight each peak direction instead
    # of thresholding the peaks based on its value
    c_p = ((evals[..., 1] - evals[..., 0]) / evals[..., 2])**args.damp

    # 4. Create histogram for each new voxel
    sphere = get_sphere('repulsion100')
    b_mat, b_inv = sh_to_sf_matrix(sphere, 8)
    sh = np.zeros(np.append(n_chunks_per_axis, b_mat.shape[0]))
    print("Creating hist fod here.")
    for chunk_x in range(n_chunks_per_axis[0]):
        for chunk_y in range(n_chunks_per_axis[1]):
            for chunk_z in range(n_chunks_per_axis[2]):
                chunk = p[chunk_x * new_voxel_size_to_vox[0]:
                          (chunk_x + 1) * new_voxel_size_to_vox[0],
                          chunk_y * new_voxel_size_to_vox[1]:
                          (chunk_y + 1) * new_voxel_size_to_vox[1],
                          chunk_z * new_voxel_size_to_vox[2]:
                          (chunk_z + 1) * new_voxel_size_to_vox[2], :]
                chunk_certainty = c_p[chunk_x * new_voxel_size_to_vox[0]:
                                      (chunk_x + 1) * new_voxel_size_to_vox[0],
                                      chunk_y * new_voxel_size_to_vox[1]:
                                      (chunk_y + 1) * new_voxel_size_to_vox[1],
                                      chunk_z * new_voxel_size_to_vox[2]:
                                      (chunk_z + 1) * new_voxel_size_to_vox[2]]
                score = np.abs(chunk.dot(sphere.vertices.T))
                ind = np.argmax(score, axis=-1).flatten()
                sf = np.zeros(sphere.vertices.shape[0])
                sf[ind] += chunk_certainty.flatten()
                sh[chunk_x, chunk_y, chunk_z, :] = sf.dot(b_inv)

    nib.save(nib.Nifti1Image(sh, np.diag(np.append([args.new_voxel_size]*3, [1.0]))),
             args.out_sh)


if __name__ == '__main__':
    main()
