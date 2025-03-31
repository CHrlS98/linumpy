#!/usr/bin/env python3
import argparse
import numpy as np
import zarr
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image')
    p.add_argument('out_image')
    p.add_argument('--threshold', type=float, default=0.0)
    p.add_argument('--sigma', type=float, default=10.0)
    p.add_argument('--zmin', type=int)
    p.add_argument('--zmax', type=int)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    im = zarr.open(args.in_image, mode='r')
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
    print(xmin, xmax, ymin, ymax)
    print(im.shape)
    out = zarr.open(args.out_image, shape=(len(zrange), xmax - xmin, ymax - ymin),
                    chunks=im.chunks, dtype=im.dtype, mode='w')
    for z in tqdm(range(len(zrange)), 'Cropping slices'):
        out[z] = im[z, xmin:xmax, ymin:ymax]


if __name__ == '__main__':
    main()
