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

from linumpy.preproc.xyzcorr import findTissueInterface
from linumpy.preproc.icorr import get_extendedAttenuation_Vermeer2013
from linumpy.io.zarr import read_omezarr, save_omezarr


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


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Loading the data
    zarr_vol, res = read_omezarr(args.input, level=0)
    attn = None
    vol = zarr_vol[:]

    # Step 1. Detect the tissue interface
    # Compute image gradient along a-lines using 1st order derivative of Gaussian
    dz = gaussian_filter1d(vol[:], sigma=args.s_z, order=1,
                               axis=0, mode='constant')
    interface_index = np.argmax(dz, axis=0)
    zz, _, _ = np.meshgrid(np.arange(dz.shape[0]),
                           np.arange(dz.shape[1]),
                           np.arange(dz.shape[2]),
                           indexing='ij')
    mask = zz >= interface_index
    vol[~mask] = 0

    import matplotlib.pyplot as plt
    all_alines = np.reshape(vol, (vol.shape[0], -1)).T
    from tqdm import tqdm
    for aline in tqdm(all_alines[:100000]):
        plt.plot(aline)
    plt.show()

    # save_omezarr(attn.astype(np.float32), args.output,
    #           voxel_size=res, chunks=zarr_vol.chunks)


if __name__ == "__main__":
    main()
