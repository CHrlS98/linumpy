#!/usr/bin/env python3
"""
Crop a 3D reconstruction volume as tightly as possible around the brain.
Lateral cropping is done by identifying the brain in each slice and cropping
to the min/max coordinates across all slices. Slices can also be removed from
the top and bottom of the volume manually by specifying zmin and zmax.
"""
import argparse
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from linumpy.io.zarr import read_omezarr, OmeZarrWriter


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

    writer = OmeZarrWriter(args.out_image,
                           shape=(len(zrange), xmax - xmin, ymax - ymin),
                           chunks=im.chunks, dtype=im.dtype)
    for z in tqdm(range(len(zrange)), 'Cropping slices'):
        writer[z] = im[z, xmin:xmax, ymin:ymax]

    writer.finalize(res, n_levels=3)


if __name__ == '__main__':
    main()
