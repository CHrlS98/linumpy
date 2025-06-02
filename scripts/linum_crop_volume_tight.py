#!/usr/bin/env python3
"""
Crop a 3D reconstruction volume as tightly as possible around the brain.
"""
import argparse
import numpy as np
import zarr
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from linumpy.io.zarr import read_omezarr, save_zarr
import dask.array as da


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image',
                   help='Input image in .ome.zarr.')
    p.add_argument('out_image',
                   help='Output image (.ome.zarr).')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Threshold for identifying background voxels [%(default)s].')
    p.add_argument('--sigma', type=float, default=10.0,
                   help='Smoothing sigma [%(default)s].')
    p.add_argument('--zmin', type=int,
                   help='Remove slices before zmin.')
    p.add_argument('--zmax', type=int,
                   help='Remove slices after zmax.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    im, res = read_omezarr(args.in_image)
    xbounds = []
    ybounds = []
    zmin = args.zmin
    zmax = args.zmax
    if zmin is None:
        zmin = 0
    if zmax is None:
        zmax = im.shape[0]

    zrange = np.arange(zmin, zmax)
    for z in tqdm(zrange, 'Extracting min/max per slice'):
        brain = gaussian_filter(im[z], args.sigma) > args.threshold
        xids, yids = np.nonzero(brain)
        xbounds.append([xids.min(), xids.max()])
        ybounds.append([yids.min(), yids.max()])

    xbounds = np.asarray(xbounds)
    ybounds = np.asarray(ybounds)
    xmin = xbounds[:, 0].min()
    xmax = xbounds[:, 1].max()
    ymin = ybounds[:, 0].min()
    ymax = ybounds[:, 1].max()

    temp_store = zarr.TempStore()
    out = zarr.open(temp_store, shape=(len(zrange), xmax - xmin, ymax - ymin),
                    chunks=im.chunks, dtype=im.dtype, mode='w')
    for z in tqdm(range(len(zrange)), 'Cropping slices'):
        out[z] = im[z, xmin:xmax, ymin:ymax]

    dask_out = da.from_zarr(out)
    save_zarr(dask_out, args.out_image, scales=res, chunks=im.chunks, n_levels=3)


if __name__ == '__main__':
    main()
