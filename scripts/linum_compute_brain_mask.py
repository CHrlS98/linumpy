#!/usr/bin/env python3
import argparse
from linumpy.io.zarr import read_omezarr, save_zarr
import zarr
import numpy as np
from skimage.morphology import disk
from skimage.segmentation import morphological_chan_vese, disk_level_set
from scipy.ndimage import gaussian_filter, center_of_mass, binary_closing

import dask.array as da
from tqdm import tqdm
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_volume')
    p.add_argument('out_mask')
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--n_iters', type=int, default=100)
    p.add_argument('--clip_min', type=int, default=2,
                   help='Minimum clip value (percentile of each slice intensities).')
    p.add_argument('--clip_max', type=int, default=98,
                   help='Maximum clip value (percentile of each slice intensities).')
    return p


def process_slice(volume, mask, index, sigma,
                  n_iters, clip_min, clip_max):
    vol = gaussian_filter(volume[index], sigma)
    vol = np.clip(vol, np.percentile(vol, clip_min), np.percentile(vol, clip_max))
    vol -= vol.min()
    vol /= vol.max()

    r_center, c_center = center_of_mass(vol)

    init_level_set = disk_level_set(vol.shape, center=(int(r_center), int(c_center)))
    segment = morphological_chan_vese(vol, n_iters, init_level_set=init_level_set,
                                      lambda2=4.0, lambda1=1.0)

    mask[index] = segment

    if index % 20 == 0:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(vol, cmap='gray')
        ax.imshow(segment, alpha=0.5, cmap='jet')
        fig.savefig(f'segment{index}.png')


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if '.ome.zarr' in args.in_volume:
        vol, res = read_omezarr(args.in_volume)
    else:
        vol = zarr.open(args.in_volume, mode='r')
    tile_shape = vol.chunks

    # assumption: all slices contain foreground and background
    temp_store = zarr.TempStore()
    foo = zarr.group(store=temp_store)
    mask = foo.empty(name='mask', shape=vol.shape,
                     dtype=np.float32, chunks=tile_shape)

    for i in tqdm(range(vol.shape[0])):
        process_slice(vol, mask, i,
                      args.sigma, args.n_iters,
                      args.clip_min, args.clip_max)
    if '.ome.zarr' in args.out_mask:
        dask_out = da.from_zarr(mask)
        save_zarr(dask_out, args.out_mask,
                scales=res, chunks=tile_shape)
    else:
        output = zarr.open(args.out_mask, mode='w', chunks=vol.chunks)
        zarr.copy(mask, output)


if __name__ == '__main__':
    main()
