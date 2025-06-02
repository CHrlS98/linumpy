#!/usr/bin/env python3
"""
Crop a 3D OME-ZARR volume to a specified range of indices.
"""
import argparse
import zarr
import dask.array as da
import numpy as np
from linumpy.io.zarr import read_omezarr, save_zarr


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_volume',
                   help='Input volume in OME-ZARR format.')
    p.add_argument('out_cropped',
                   help='Output volume in OME-ZARR format.')
    p.add_argument('--start_index', type=int, nargs=3, required=True,
                   help='Start index for cropping (x, y, z).')
    p.add_argument('--stop_index', type=int, nargs=3, required=True,
                   help='Stop index for cropping (x, y, z).')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Read the input volume
    vol, res = read_omezarr(args.in_volume)

    # Crop the volume using the specified indices
    cropped_vol = vol[args.start_index[0]:args.stop_index[0],
                      args.start_index[1]:args.stop_index[1],
                      args.start_index[2]:args.stop_index[2]]

    # Save the cropped volume to the output path
    dask_arr = da.from_array(cropped_vol)
    save_zarr(dask_arr, args.out_cropped, scales=res,
              chunks=vol.chunks,
              n_levels=3)
    

if __name__ == '__main__':
    main()
