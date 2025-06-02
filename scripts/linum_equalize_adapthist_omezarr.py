#!/usr/bin/env python3
from skimage.exposure import equalize_adapthist
import argparse
import numpy as np
from linumpy.io.zarr import read_omezarr, save_zarr
import zarr
import dask.array as da


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_zarr",
                   help="Full path to a zarr file.")
    p.add_argument("out_zarr",
                   help="Full path to the output zarr file")
    p.add_argument("--clip_limit", type=float, default=0.01,
                   help="Clip limit for CLAHE. [%(default)s]")
    p.add_argument("--nbins", type=int, default=256,
                   help="Number of bins for histogram. [%(default)s]")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image, res = read_omezarr(args.in_zarr)

    out = equalize_adapthist(np.asarray(image),
                             clip_limit=args.clip_limit,
                             nbins=args.nbins)

    da_out = da.from_array(out)
    save_zarr(da_out, args.out_zarr, scales=res,
              chunks=image.chunks, n_levels=3)


if __name__ == '__main__':
    main()