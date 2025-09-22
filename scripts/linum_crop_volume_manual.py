#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop a ome.zarr file manually from upper/lower bounds along each axis.
"""
import argparse
from linumpy.io import save_omezarr, read_omezarr

import dask.array as da


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_volume')
    p.add_argument('out_volume')
    p.add_argument('--xmin', default=0, type=int)
    p.add_argument('--xmax', default=-1, type=int)
    p.add_argument('--ymin', default=0, type=int)
    p.add_argument('--ymax', default=-1, type=int)
    p.add_argument('--zmin', default=0, type=int)
    p.add_argument('--zmax', default=-1, type=int)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_volume)
    darr = da.from_zarr(vol)
    darr = darr[args.xmin:args.xmax,
                args.ymin:args.ymax,
                args.zmin:args.zmax]
    save_omezarr(darr, args.out_volume, res, vol.chunks)


if __name__ == '__main__':
    main()
