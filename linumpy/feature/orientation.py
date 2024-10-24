# -*- coding:utf8 -*-
from functools import partial
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from scipy.ndimage import correlate
import numpy as np
from tqdm import tqdm
from linumpy.io.zarr import create_directory
import zarr
import dask.array as da


def _gaussian_1d(r):
    out = np.exp(-r**2, dtype=np.float64)
    return out


def _gaussian_2d(r1, r2):
    out = np.exp(-r1**2, dtype=np.float64)*\
          np.exp(-r2**2, dtype=np.float64)
    return out


def _make_xfilter(f):
    out = np.reshape(f, (-1, 1, 1))
    return out


def _make_yfilter(f):
    out = np.reshape(f, (1, -1, 1))
    return out


def _make_zfilter(f):
    out = np.reshape(f, (1, 1, -1))
    return out


def _make_xyfilter(f):
    _sx, _sy = f.shape
    out = np.reshape(f, (_sx, _sy, 1))
    return out


def _make_xzfilter(f):
    _sx, _sz = f.shape
    out = np.reshape(f, (_sx, 1, _sz))
    return out


def _make_yzfilter(f):
    _sy, _sz = f.shape
    out = np.reshape(f, (1, _sy, _sz))
    return out


def _kappa_fct(alpha, beta, gamma, coeff=1.0, pow_alpha=0.0,
               pow_beta=0.0, pow_gamma=0.0):
    out = coeff*alpha**pow_alpha*beta**pow_beta*gamma**pow_gamma
    out = np.reshape(out, (1, 1, 1, -1))
    return out.astype(np.float32)


def equalize_filter(wfilter):
    sum_pos = np.sum(wfilter[wfilter > 0])
    sum_neg = np.sum(np.abs(wfilter[wfilter < 0]))
    wfilter_eq = wfilter
    wfilter_eq[wfilter < 0] = wfilter[wfilter < 0] / sum_neg * sum_pos
    return wfilter_eq


def convolve_with_bank(image, filter_bank, norm=1.0, mode='reflect'):
    # the sum of negatives must equal the sum of positives
    out = image * norm
    for wfilter in filter_bank:
        wfilter_eq = equalize_filter(wfilter)
        out = correlate(out, wfilter_eq, mode=mode)
    return out


def _data_to_mmap(data, dir, fname, datatype):
    data_fp = np.memmap(os.path.join(dir, fname), dtype=datatype,
                        mode='w+', shape=data.shape)
    data_fp[:] = data.astype(datatype)[:]
    data_fp.flush()
    return data_fp


class Steerable4thOrderGaussianQuadratureFilter():
    def __init__(self, image, R, store_path, mode='reflect'):
        argtol = 3.0  # np.sqrt(-np.log(rtol))
        samples = np.linspace(0, argtol, R+1)
        samples = np.append(-samples[:0:-1], samples)
        self.image_shape = image.shape
        self.samples = samples

        # additional normalization constant to generate 0-area-under-curve G filters
        self._C_forGFilters = 4.0*np.sqrt(210)/(105*np.sqrt(np.pi))

        self._N_2Dto3D = np.float_power(2.0/np.pi, 0.25).astype(np.float64)
        self._sampling_delta = np.float_power(argtol / R, 3)

        self._g4_funcs_list = [
            self._g4a, self._g4b, self._g4c, self._g4d, self._g4e, self._g4f,
            self._g4g, self._g4h, self._g4i, self._g4j, self._g4k, self._g4l,
            self._g4m, self._g4n, self._g4o
        ]
        self._h4_funcs_list = [
            self._h4a, self._h4b, self._h4c, self._h4d, self._h4e, self._h4f,
            self._h4g, self._h4h, self._h4i, self._h4j, self._h4k, self._h4l,
            self._h4m, self._h4n, self._h4o, self._h4p, self._h4q, self._h4r,
            self._h4s, self._h4t, self._h4u
        ]

        self._g4_kappas = []
        self._h4_kappas = []

        self.is_g_response_computed = False
        self.is_h_response_computed = False

        create_directory(store_path, overwrite=True)
        self.zarr_store = zarr.DirectoryStore(path=store_path)
        self.zarr_root = zarr.open(self.zarr_store, mode='w')

        self.chunkshape = (80, 80, 80)
        self.blocks_dims = (int(float(image.shape[0]) / self.chunkshape[0] + 0.5),
                            int(float(image.shape[1]) / self.chunkshape[1] + 0.5),
                            int(float(image.shape[2]) / self.chunkshape[2] + 0.5))
        print(self.blocks_dims)

        for it, _g_func in enumerate(tqdm(self._g4_funcs_list)):
            _kappa, _filters = _g_func(samples)
            self._g4_kappas.append(_kappa)
            data = convolve_with_bank(image, _filters, self._N_2Dto3D * self._sampling_delta, mode)
            self.zarr_root.array(f'g4_{it}', data[..., None],
                                 chunks=self.chunkshape + (1,),
                                 dtype=np.float32,
                                 write_empty_chunks=False)

        for it, _h_func in enumerate(tqdm(self._h4_funcs_list)):
            _kappa, _filters = _h_func(samples)
            self._h4_kappas.append(_kappa)
            data = convolve_with_bank(image, _filters, self._N_2Dto3D * self._sampling_delta, mode)
            self.zarr_root.array(f'h4_{it}', data[..., None],
                                 chunks=self.chunkshape + (1,),
                                 dtype=np.float32,
                                 write_empty_chunks=False)

    def compute_quadrature_output(self, directions):
        print('Compute quadrature response')
        out_zarr = self.zarr_root.zeros('q_response', shape=self.image_shape + (len(directions),),
                                        chunks=self.chunkshape + (10,), dtype=np.float32)

        if not self.is_g_response_computed:
            self.compute_G_response(directions)
        if not self.is_h_response_computed:
            self.compute_H_response(directions)

        futures = []
        with ProcessPoolExecutor(max_workers=18) as executor:
            for (i, j, k) in product(*[range(_i) for _i in self.blocks_dims]):
                futures.append(executor.submit(self.quadrature_response, i, j, k))
            for f in as_completed(futures):
                f.result()  # this step for throwing exceptions in main thread

        print('done.')
        return out_zarr

    def compute_G_response(self, directions):
        out_zarr = self.zarr_root.zeros('g_response', shape=self.image_shape + (len(directions),),
                                        chunks=self.chunkshape + (10,), dtype=np.float32)

        print('Interpolate feature maps for G response')
        # Interpolate feature maps in blocks
        for it, _kappa in enumerate(tqdm(self._g4_kappas)):
            alpha = directions[:, 0]
            beta = directions[:, 1]
            gamma = directions[:, 2]
            with ProcessPoolExecutor(max_workers=18) as executor:
                futures = {}
                for (i, j, k) in product(*[range(_i) for _i in self.blocks_dims]):
                    futures[executor.submit(
                        self.add_response, 'g',
                        _kappa(alpha, beta, gamma),
                        it, i, j, k)] = (i, j, k)
                for f in as_completed(futures):
                    f.result()

        # Square the resulting image
        with ProcessPoolExecutor(max_workers=18) as executor:
                futures = {}
                for (i, j, k) in product(*[range(_i) for _i in self.blocks_dims]):
                    futures[executor.submit(
                        self.square_response,
                        'g', i, j, k)] = (i, j, k)
                for f in as_completed(futures):
                    f.result()

        self.is_g_response_computed = True
        # output zarr image
        return out_zarr

    def compute_H_response(self, directions):
        out_zarr = self.zarr_root.zeros('h_response', shape=self.image_shape + (len(directions),),
                                        chunks=self.chunkshape + (10,), dtype=np.float32)

        print('Interpolate feature maps for H response')
        # Interpolate feature maps in blocks
        for it, _kappa in enumerate(tqdm(self._h4_kappas)):
            alpha = directions[:, 0]
            beta = directions[:, 1]
            gamma = directions[:, 2]
            with ProcessPoolExecutor(max_workers=18) as executor:
                futures = {}
                for (i, j, k) in product(*[range(_i) for _i in self.blocks_dims]):
                    futures[executor.submit(
                        self.add_response, 'h',
                        _kappa(alpha, beta, gamma),
                        it, i, j, k)] = (i, j, k)
                for f in as_completed(futures):
                    f.result()

        # Square the resulting image
        with ProcessPoolExecutor(max_workers=18) as executor:
                futures = {}
                for (i, j, k) in product(*[range(_i) for _i in self.blocks_dims]):
                    futures[executor.submit(
                        self.square_response,
                        'h', i, j, k)] = (i, j, k)
                for _ in as_completed(futures):
                    f.result()

        self.is_h_response_computed = True
        # output zarr image
        return out_zarr

    def add_response(self, key, kappa_arr, it, i, j, k):
        h_response = self.zarr_root[f'{key}_response']
        h4_it = self.zarr_root[f'{key}4_{it}']
        h_response.blocks[i, j, k] += kappa_arr * h4_it.blocks[i, j, k]

    def square_response(self, key, i, j, k):
        h_response = self.zarr_root[f'{key}_response']
        h_response.blocks[i, j, k] = h_response.blocks[i, j, k]**2

    def quadrature_response(self, i, j, k):
        g_response = self.zarr_root['g_response']
        h_response = self.zarr_root['h_response']
        q_response = self.zarr_root['q_response']
        q_response.blocks[i, j, k] =\
            g_response.blocks[i, j, k] +\
                h_response.blocks[i, j, k]

    def _g4a(self, r):
        kappa = partial(_kappa_fct, pow_gamma=4)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter(self._C_forGFilters*(4.0*r**4 - 12.0*r**2 + 3.0)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4b(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_alpha=1.0, pow_gamma=3.0)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*_gaussian_1d(r))
        zfilter = _make_zfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4c(self, r):
        kappa = partial(_kappa_fct, coeff=6.0, pow_alpha=2, pow_gamma=2)
        xfilter = _make_xfilter((2*r**2 - 1)*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*_gaussian_1d(r))
        zfilter = _make_zfilter((2*r**2 - 1)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4d(self, r):
        kappa = partial(_kappa_fct, coeff=4.0, pow_alpha=3, pow_gamma=1)
        xfilter = _make_xfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4e(self, r):
        kappa = partial(_kappa_fct, pow_alpha=4)
        xfilter = _make_xfilter((4.0*r**4 - 12.0*r**2 + 3.0)*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4f(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_beta=1, pow_gamma=3)
        xfilter = _make_xfilter(self._C_forGFilters*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4g(self, r):
        kappa = partial(_kappa_fct, coeff=12, pow_alpha=1, pow_beta=1, pow_gamma=2)
        xfilter = _make_xfilter(self._C_forGFilters*r*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter((4.0*r**2 - 2.0)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4h(self, r):
        kappa = partial(_kappa_fct, coeff=12, pow_alpha=2, pow_beta=1, pow_gamma=1)
        xfilter = _make_xfilter((4.0*r**2 - 2.0)*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(self._C_forGFilters*r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4i(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_alpha=3, pow_beta=1)
        xfilter = _make_xfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        yfilter = _make_yfilter(self._C_forGFilters*r*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4j(self, r):
        kappa = partial(_kappa_fct, coeff=6, pow_beta=2, pow_gamma=2)
        xfilter = _make_xfilter(self._C_forGFilters*_gaussian_1d(r))
        yfilter = _make_yfilter((2*r**2 - 1)*_gaussian_1d(r))
        zfilter = _make_zfilter((2*r**2 - 1)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4k(self, r):
        kappa = partial(_kappa_fct, coeff=12, pow_alpha=1, pow_beta=2, pow_gamma=1)
        xfilter = _make_xfilter(self._C_forGFilters*r*_gaussian_1d(r))
        yfilter = _make_yfilter((4.0*r**2 - 2.0)*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4l(self, r):
        kappa = partial(_kappa_fct, coeff=6, pow_alpha=2, pow_beta=2)
        xfilter = _make_xfilter((2*r**2 - 1)*_gaussian_1d(r))
        yfilter = _make_yfilter((2*r**2 - 1)*_gaussian_1d(r))
        zfilter = _make_zfilter(self._C_forGFilters*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4m(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_beta=3, pow_gamma=1)
        xfilter = _make_xfilter(self._C_forGFilters*_gaussian_1d(r))
        yfilter = _make_yfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4n(self, r):
        kappa = partial(_kappa_fct, coeff=4, pow_alpha=1, pow_beta=3)
        xfilter = _make_xfilter(self._C_forGFilters*r*_gaussian_1d(r))
        yfilter = _make_yfilter(r*(4.0*r**2 - 6.0)*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _g4o(self, r):
        kappa = partial(_kappa_fct, pow_beta=4)
        xfilter = _make_xfilter(self._C_forGFilters*_gaussian_1d(r))
        yfilter = _make_yfilter((4.0*r**4 - 12.0*r**2 + 3.0)*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4a(self, r):
        kappa = partial(_kappa_fct, pow_gamma=5)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter(r*(0.3975*r**4 - 2.982*r**2 + 2.858)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4b(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_alpha=1.0, pow_gamma=4.0)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4c(self, r):
        _x, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_alpha=2, pow_gamma=3)
        # this filter is not x-z separable, so the output is a 2D filter
        xzfilter = _make_xzfilter(_z*(0.3975*_x**2*_z**2 - 0.8946*_x**2 -0.2982*_z**2 + 0.5716)*_gaussian_2d(_x, _z))
        yfilter = _make_yfilter(_gaussian_1d(r))
        return kappa, (xzfilter, yfilter)

    def _h4d(self, r):
        _x, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_alpha=3, pow_gamma=2)
        xzfilter = _make_xzfilter(_x*(0.3975*_x**2*_z**2 - 0.8946*_z**2 -0.2982*_x**2 + 0.5716)*_gaussian_2d(_x, _z))
        yfilter = _make_yfilter(_gaussian_1d(r))
        return kappa, (xzfilter, yfilter)

    def _h4e(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_alpha=4, pow_gamma=1)
        xfilter = _make_xfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4f(self, r):
        kappa = partial(_kappa_fct, pow_alpha=5)
        xfilter = _make_xfilter(r*(0.3975*r**4 - 2.982*r**2 + 2.858)*_gaussian_1d(r))
        yfilter = _make_yfilter(_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4g(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_beta=1, pow_gamma=4)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4h(self, r):
        kappa = partial(_kappa_fct, coeff=20, pow_alpha=1, pow_beta=1, pow_gamma=3)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(r*(0.3975*r**2 - 0.8946)*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4i(self, r):
        _x, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=30, pow_alpha=2, pow_beta=1, pow_gamma=2)
        xzfilter = _make_xzfilter((0.3975*_x**2*_z**2 - 0.2982*_x**2 - 0.2982*_z**2 + 0.19053)*_gaussian_2d(_x, _z))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        return kappa, (xzfilter, yfilter)

    def _h4j(self, r):
        kappa = partial(_kappa_fct, coeff=20, pow_alpha=3, pow_beta=1, pow_gamma=1)
        xfilter = _make_xfilter(r*(0.3975*r**2 - 0.8946)*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4k(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_alpha=4, pow_beta=1)
        xfilter = _make_xfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        yfilter = _make_yfilter(r*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4l(self, r):
        _y, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_beta=2, pow_gamma=3)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yzfilter = _make_yzfilter(_z*(0.3975*_y**2*_z**2 - 0.8946*_y**2 - 0.2982*_z**2 + 0.5716)*_gaussian_2d(_y, _z))
        return kappa, (xfilter, yzfilter)

    def _h4m(self, r):
        _y, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=30, pow_alpha=1, pow_beta=2, pow_gamma=2)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yzfilter = _make_yzfilter((0.3975*_z**2*_y**2 - 0.2982*_z**2 - 0.2982*_y**2 + 0.19053)*_gaussian_2d(_y, _z))
        return kappa, (xfilter, yzfilter)

    def _h4n(self, r):
        _x, _y = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=30, pow_alpha=2, pow_beta=2, pow_gamma=1)
        xyfilter = _make_xyfilter((0.3975*_x**2*_y**2 - 0.2982*_x**2 - 0.2982*_y**2 + 0.19053)*_gaussian_2d(_x, _y))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xyfilter, zfilter)

    def _h4o(self, r):
        _x, _y = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_alpha=3, pow_gamma=2)
        xyfilter = _make_xyfilter(_x*(0.3975*_x**2*_y**2 - 0.2982*_x**2 - 0.8946*_y**2 + 0.5716)*_gaussian_2d(_x, _y))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xyfilter, zfilter)

    def _h4p(self, r):
        _y, _z = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_beta=3, pow_gamma=2)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yzfilter = _make_yzfilter(_y*(0.3975*_y**2*_z**2 - 0.2982*_y**2 - 0.8946*_z**2 + 0.5716)*_gaussian_2d(_y, _z))
        return kappa, (xfilter, yzfilter)

    def _h4q(self, r):
        kappa = partial(_kappa_fct, coeff=20, pow_alpha=1, pow_beta=3, pow_gamma=1)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter(r*(0.3975*r**2 - 0.8946)*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4r(self, r):
        _x, _y = np.meshgrid(r, r, indexing='ij')
        kappa = partial(_kappa_fct, coeff=10, pow_alpha=2, pow_beta=3)
        xyfilter = _make_xyfilter(_y*(0.3975*_x**2*_y**2 - 0.8946*_x**2 - 0.2982*_y**2 + 0.5716)*_gaussian_2d(_x, _y))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xyfilter, zfilter)

    def _h4s(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_beta=4, pow_gamma=1)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        zfilter = _make_zfilter(r*_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4t(self, r):
        kappa = partial(_kappa_fct, coeff=5, pow_alpha=1, pow_beta=4)
        xfilter = _make_xfilter(r*_gaussian_1d(r))
        yfilter = _make_yfilter((0.3975*r**4 - 1.7892*r**2 + 0.5716)*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)

    def _h4u(self, r):
        kappa = partial(_kappa_fct, pow_beta=5)
        xfilter = _make_xfilter(_gaussian_1d(r))
        yfilter = _make_yfilter(r*(0.3975*r**4 - 2.982*r**2 + 2.858)*_gaussian_1d(r))
        zfilter = _make_zfilter(_gaussian_1d(r))
        return kappa, (xfilter, yfilter, zfilter)
