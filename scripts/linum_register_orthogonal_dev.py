#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import zarr
import numpy as np
from tqdm import tqdm

from scipy.ndimage import map_coordinates, shift
from skimage.filters import threshold_otsu
from scipy.optimize import minimize
from skimage.transform import pyramid_gaussian, pyramid_laplacian
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('fixed')
    p.add_argument('moving')
    p.add_argument('fixed_z', type=int)
    p.add_argument('moving_z', type=int)
    p.add_argument('-r', '--resolution', type=float,
                   default=(200, 1.875, 1.875), nargs=3,
                   help='Resolution in [z, y, x] order in microns.')
    return p


def get_sample_coordinates(tx, ty, theta, step_size, n_points):
    origin = np.array([tx, ty])
    direction = np.array([np.cos(theta), np.sin(theta)])
    forward = [origin]
    n = 1
    while n < n_points:
        forward.append(forward[-1] + step_size*direction/np.linalg.norm(direction))
        n += 1
    # backward = [origin]
    # while 0 <= backward[-1][0] < bounds[0] and 0 <= backward[-1][1] < bounds[1]:
    #     backward.append(backward[-1] - step_size*direction/np.linalg.norm(direction))
    return np.asarray(forward)


def objective_function(params, I_a, I_b):
    t1x = params[0]
    t1y = params[1]
    theta1 = params[2]
    t2x = params[3]
    t2y = params[4]
    theta2 = params[5]

    samples_a = get_sample_coordinates(t1x, t1y, theta1, 1, 600)
    samples_b = get_sample_coordinates(t2x, t2y, theta2, 1, 600)

    line_a = map_coordinates(I_a, samples_a.T, mode='nearest')
    line_b = map_coordinates(I_b, samples_b.T, mode='nearest')

    return np.sum((line_a - line_b)**2)


def jacobian(params, I_a, I_b, dIa_dx, dIa_dy, dIb_dx, dIb_dy):
    t1x = params[0]
    t1y = params[1]
    theta1 = params[2]
    t2x = params[3]
    t2y = params[4]
    theta2 = params[5]

    samples_a = get_sample_coordinates(t1x, t1y, theta1, 1, 600)
    samples_b = get_sample_coordinates(t2x, t2y, theta2, 1, 600)

    term0 = 2.0*(map_coordinates(I_a, samples_a.T) - map_coordinates(I_b, samples_b.T))
    dt1x = np.sum(term0*map_coordinates(dIa_dx, samples_a.T))
    dt1y = np.sum(term0*map_coordinates(dIa_dy, samples_a.T))
    dtheta1 = np.sum(term0*(map_coordinates(dIa_dx, samples_a.T)*(-samples_a[:, 0]*np.sin(theta1)-samples_a[:, 1]*np.cos(theta1)) +
                            map_coordinates(dIa_dy, samples_a.T)*(samples_a[:, 0]*np.cos(theta1)-samples_a[:, 1]*np.sin(theta1))))
    dt2x = np.sum(term0*map_coordinates(dIb_dx, samples_b.T))
    dt2y = np.sum(term0*map_coordinates(dIb_dy, samples_b.T))
    dtheta2 = np.sum(term0*(map_coordinates(dIb_dx, samples_b.T)*(-samples_b[:, 0]*np.sin(theta2)-samples_b[:, 1]*np.cos(theta2)) +
                            map_coordinates(dIb_dy, samples_b.T)*(samples_b[:, 0]*np.cos(theta2)-samples_b[:, 1]*np.sin(theta2))))

    out = np.array([
        dt1x, dt1y, dtheta1, dt2x, dt2y, dtheta2
    ])
    print(out)
    return out


def apply_transform(I, tx, ty, theta):
    cx, cy = np.meshgrid(np.arange(I.shape[0]), np.arange(I.shape[1]), indexing='ij')
    coordinates = np.stack([cx*np.cos(theta) - cy*np.sin(theta) + tx,
                            cx*np.sin(theta) + cy*np.cos(theta) + ty],
                            axis=0)
    return map_coordinates(I, coordinates)


def objective_ssd(params, I, J):
    tx = params[0]
    ty = params[1]
    rtheta = params[2]
    J_prime = apply_transform(J, tx, ty, rtheta)
    return np.sum((I - J_prime)**2)


def _make_common_last_axis(images):
    N = max([i.shape[1] for i in images])
    padded = []
    for i in images:
        i_padded = np.zeros((i.shape[0], N), dtype=i.dtype)
        i_padded[:i.shape[0], :i.shape[1]] = i
        padded.append(i_padded)
    return padded


def correlation_coefficient(I_a, I_b):
    mu_a = np.mean(I_a)
    mu_b = np.mean(I_b)
    p = np.sum((I_a - mu_a)*(I_b - mu_b)) / (np.sqrt(np.sum((I_a - mu_a)**2))*np.sqrt(np.sum((I_b - mu_b)**2)))
    return p


def mutual_information(I_a, I_b):
    vmin = min(np.min(I_a), np.min(I_b))
    vmax = max(np.max(I_a), np.max(I_b))
    hist_ab, _, _ = np.histogram2d(I_a, I_b, bins=10, range=[(vmin, vmax), (vmin, vmax)])
    hist_ab /= np.sum(hist_ab)
    hist_a = np.sum(hist_ab, axis=1)
    hist_b = np.sum(hist_ab, axis=0)
    hist_a_x_hist_b = hist_a.reshape((-1, 1))*hist_b.reshape((1, -1))
    if (hist_a_x_hist_b == 0).any():
        return -np.inf
    MI = np.sum(hist_ab*np.log(hist_ab/hist_a_x_hist_b))
    return MI


def gen_overlay_image(index, ref_image, axis):
    mask = np.zeros(ref_image.shape, dtype=bool)
    if axis == 0:
        mask[index, :] = True
    elif axis == 1:
        mask[:, index] = True
    else:
        raise ValueError(f"Invalid axis {axis}")
    overlay = np.ma.masked_array(ref_image, ~mask)

    return overlay


def _make_common_shape(a, b):
    M = max(a.shape[0], b.shape[0])
    N = max(a.shape[1], b.shape[1])
    a_padded = np.pad(a, ((0, M - a.shape[0]), (0, N - a.shape[1])))
    b_padded = np.pad(b, ((0, M - b.shape[0]), (0, N - b.shape[1])))
    return a_padded, b_padded


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    fixed_zarr = zarr.open(args.fixed, mode='r')
    moving_zarr = zarr.open(args.moving, mode='r')

    some_z_coronal = moving_zarr.shape[0] // 2
    coronal_full_resolution = moving_zarr[some_z_coronal].T

    conv_200um_in_px = int(np.ceil(200.0 / 1.875))

    x, y = np.meshgrid(np.arange(0, fixed_zarr.shape[0], 1.875/200.0),
                       np.arange(fixed_zarr.shape[2]),
                       indexing='ij')
    xy = np.stack([x, y], axis=0)

    error_curve = []
    best_z = 0
    best_error = np.inf
    for z in tqdm(range(fixed_zarr.shape[1] // conv_200um_in_px)):
        coronal_from_axial = fixed_zarr[:, z*conv_200um_in_px:(z+1)*conv_200um_in_px, :]
        coronal_from_axial = np.mean(coronal_from_axial, axis=1)
        coronal_from_axial = map_coordinates(coronal_from_axial, xy)
        a, b = _make_common_shape(coronal_from_axial, coronal_full_resolution)
        res = minimize(lambda x: objective_ssd(x, a, b), np.array([0, 0, 0]))
        b_prime = apply_transform(b, res.x[0], res.x[1], res.x[2])
        error = np.sum((a - b_prime)**2)
        if error < best_error:
            best_error = error
            best_z = z*conv_200um_in_px
            print(best_error)
        error_curve.append(error)

    plt.plot(error_curve)
    plt.show()


def _mainx():
    parser = _build_arg_parser()
    args = parser.parse_args()

    fixed_zarr = zarr.open(args.fixed, mode='r')
    moving_zarr = zarr.open(args.moving, mode='r')

    # fit a single image to multiple slices
    n_slices = 3
    n_skip = 6
    n_subdivides = 3
    images_a = []
    for i in range(n_slices):
        image_a_i = fixed_zarr[args.fixed_z + n_skip*i]
        image_a_i = np.abs(tuple(pyramid_gaussian(image_a_i, max_layer=n_subdivides))[-1])
        image_a_i /= image_a_i.max()
        images_a.append(image_a_i)

    image_b = moving_zarr[args.moving_z]
    image_b = image_b.T[::-1]
    image_b = np.abs(tuple(pyramid_gaussian(image_b, max_layer=n_subdivides))[-1])
    image_b /= image_b.max()

    # 200 um / (1.875 um * 2**n_subdivides)
    slices_gap = args.resolution[0] / args.resolution[1] / 2.0**n_subdivides * n_skip

    all_images = images_a + [image_b,]
    all_images = _make_common_last_axis(images_a + [image_b,])
    images_a = all_images[:-1]
    image_b = all_images[-1]

    n_scans_along_axial = images_a[0].shape[0]
    n_scans_along_coronal = image_b.shape[0]
    print('n_scans_along_coronal:', n_scans_along_coronal)

    # pad to be able to scan three slices width everywhere
    pad_width = int(np.ceil((n_slices - 1)*slices_gap))
    image_b = np.pad(image_b, ((pad_width, pad_width), (0, 0)), mode='edge')

    match = []
    for i in tqdm(range(n_scans_along_axial)):
        match.append([])
        for j in range(n_scans_along_coronal):
            m = 0
            for k in range(len(images_a)):
                image_b_k = image_b[j + int((len(images_a) - k - 1)*slices_gap)]
                m += np.max(np.correlate(images_a[k][i], image_b_k, mode='full'))
            match[-1].append(m)

    match = np.asarray(match)
    n_best = 3
    best_i, best_j = np.unravel_index(np.argsort(match, axis=None)[:-n_best-1:-1], match.shape)
    print(best_i, best_j)

    fig, ax = plt.subplots(1, n_slices + 1)
    for i in range(n_slices):
        ax[i].imshow(images_a[i], origin='lower', cmap='gray')
        ax[i].imshow(gen_overlay_image(best_i[0], images_a[i], 0), origin='lower',
                    cmap='viridis')
        ax[i].set_title(f'image_a_{i}')

    ax[n_slices].imshow(image_b, origin='lower', cmap='gray')
    for i in range(n_slices):
        ax[n_slices].imshow(gen_overlay_image(best_j[0] + int(np.ceil(i*slices_gap)), image_b, 0),
                            origin='lower', cmap='viridis')
    ax[n_slices].set_title(f'coronal {args.moving_z}')
    plt.show()

    # res = 1.875 * sampling

    # dIa_dx = np.diff(image_a, axis=0, prepend=0)
    # dIa_dy = np.diff(image_a, axis=1, prepend=0)
    # dIb_dx = np.diff(image_b, axis=0, prepend=0)
    # dIb_dy = np.diff(image_b, axis=1, prepend=0)

    # t1x = 200
    # t1y = 0
    # theta_1 = np.pi/2
    # t2x = 250
    # t2y = 0
    # theta_2 = np.pi/2
    # x0 = np.array([t1x, t1y, theta_1, t2x, t2y, theta_2])

    # plt.ion()

    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(image_a, origin='lower')
    # line_samples_a, = ax[0, 0].plot([0], [0], color='red')
    # ax[0, 1].imshow(image_b, origin='lower')
    # line_samples_b, = ax[0, 1].plot([0], [0], color='red')
    # line_amplitudes_a, = ax[1, 0].plot(np.arange(600), np.random.rand(600) * 2)
    # line_amplitudes_b, = ax[1, 1].plot(np.arange(600), np.random.rand(600) * 2)

    # def callback_wrapper(intermediate_result):
    #     return optimize_callback(intermediate_result, fig, line_samples_a, line_samples_b,
    #                              line_amplitudes_a, line_amplitudes_b, image_a, image_b)

    # res = minimize(lambda x: objective_function(x, image_a, image_b), x0,
    #                jac = lambda x: jacobian(x, image_a, image_b, dIa_dx, dIa_dy, dIb_dx, dIb_dy),
    #                callback=callback_wrapper,
    #                method='BFGS', options={'disp': True})

    # fig, ax = plt.subplots(2, 2)
    # plt.show()


if __name__ == '__main__':
    main()
