#!/usr/bin/env python 3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from linumpy.io.zarr import read_omezarr


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_zarr",
                   help="Full path to a zarr file.")
    p.add_argument("slice")
    p.add_argument("out_figure",
                   help="Full path to the output figure")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image, res = read_omezarr(args.in_zarr)


if __name__ == '__main__':
    main()
