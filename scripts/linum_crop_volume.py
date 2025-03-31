#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from linumpy.io.zarr import read_omezarr, save_zarr
from scipy.ndimage import gaussian_filter1d
import zarr
import dask.array as da

def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_volume',
                   help='Input volume in omezarr format.')
    p.add_argument('out_masked')
    p.add_argument('--sigma', type=float, default=2.0,
                   help='Smoothing sigma for mean intensity along a-line.')
    p.add_argument('--n_levels', type=int, default=5)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_volume)
    tile_shape = vol.chunks

    mean_intensity = np.mean(vol, axis=(1,2))

    sigma = 2.0
    mean_intensity = gaussian_filter1d(mean_intensity, sigma)

    fig, ax = plt.subplots(1, 1)
    # ax.plot(mean_intensity, label='Average intensity')
    d1x = np.diff(mean_intensity, 1)

    ax.plot(d1x, label='first derivative')
    ax.legend()
    fig.savefig("depth_intensity.png")
    plt.clf()

    # find start index
    start_index = np.argmax(d1x)
    stop_index = -1
    while d1x[stop_index] >= 0.0:
        stop_index -= 1

    new_shape = (vol.shape[0] + stop_index - start_index,
                 vol.shape[1],
                 vol.shape[2])
    temp_store = zarr.TempStore()
    vol_crop = zarr.open(temp_store, mode="w", shape=new_shape,
                           dtype=np.float32, chunks=tile_shape)
    vol_crop[:] = vol[start_index:stop_index]

    mean_intensity = np.mean(vol_crop, axis=(1, 2))
    plt.plot(mean_intensity)
    plt.savefig('Mean_intensity_crop.png')

    dask_arr = da.from_zarr(vol_crop)
    save_zarr(dask_arr, args.out_masked, scales=res, chunks=tile_shape,
              n_levels=args.n_levels)


if __name__ == '__main__':
    main()
