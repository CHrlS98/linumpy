#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Computes the tissue apparent attenuation coefficient map
and then use the average attenuation to compensate its effect in
the OCT reflectivity data.
"""
# TODO: Keep the OCT pixel format (which is float32 ?)
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from linumpy.preproc.xyzcorr import findTissueInterface
from linumpy.preproc.icorr import get_extendedAttenuation_Vermeer2013
from linumpy.io.zarr import read_omezarr, save_omezarr

import matplotlib.pyplot as plt

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # Mandatory parameters
    p.add_argument("input",
                   help="A single slice to process (ome-zarr).")
    p.add_argument("output",
                   help="Output attenuation map (ome-zarr).")

    # Optional argument
    p.add_argument("-m", "--mask", default=None,
                   help="Optional tissue mask (.ome.zarr)")
    p.add_argument("--s_xy", default=0.0, type=float,
                   help="Lateral smoothing sigma (default=%(default)s)")
    p.add_argument("--s_z", default=5.0, type=float,
                   help="Axial smoothing sigma (default=%(default)s)")

    return p


def exponential_decay(x, n0, lambd):
    return n0 * np.exp(-lambd * x)


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Loading the data
    zarr_vol, res = read_omezarr(args.input, level=0)
    vol = zarr_vol[:]

    contains_tissue = np.sum(vol, axis=(1, 2)) > 0
    true_depth = vol.shape[0] - np.flatnonzero(np.cumsum(contains_tissue[::-1], dtype=int) == 1)[0]

    # remove bias such that the amplitude at the end of each a-line equals 0
    vol = vol - np.min(vol[:true_depth], axis=0)

    # echoes can result in voxels with 0 intensity on top slice.
    vol = np.clip(vol, 0, None)

    # TODO: compute attenuation using the Vermeer 2013 method
    attenuation = np.zeros_like(vol)

    # u[i] = log( 1 + I[i] / sum_{j = i+1}^{true_depth}I[j])
    sum_j = np.roll(np.cumsum(gaussian_filter1d(vol[::-1], axis=0, sigma=0.5), axis=0)[::-1], -1, axis=0)
    sum_j[-1] = 0
    
    # a-lines should be processed independently
    attenuation[sum_j > 0] = 1 / (20 * res[0]) * np.log(1 + vol[sum_j > 0] / sum_j[sum_j > 0])  # cm^-1

    save_omezarr(attenuation.astype(np.float32), args.output,
                 voxel_size=res, chunks=zarr_vol.chunks)


if __name__ == "__main__":
    main()
