#!/usr/bin/env python3
import argparse
import zarr
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from tqdm import tqdm


# TODO: Robustify


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_stack',
                   help='2.5D stack in zarr format.')
    p.add_argument('out_stack',
                   help='Normalized 2.5D stack.')
    p.add_argument('--reference_index', type=int, default=0)
    p.add_argument('--at', type=float, default=0.0,
                   help='Absolute threshold [%(default)s].')
    p.add_argument('--nbins', type=int, default=100)
    return p


def eqhist(reference, reference_mask, target, at, nbins):
    target_mask = target > at
    target = np.clip(target, 0.0, None)
    target = np.floor(target / target.max() * nbins)
    matched_target = np.zeros_like(target)
    matched_flat = match_histograms(target[target_mask], reference[reference_mask])
    matched_target[target_mask] = matched_flat
    return matched_target


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_zarr = zarr.open(args.in_stack, mode='r')
    out_zarr = zarr.open(args.out_stack, mode='a',
                         shape=in_zarr.shape,
                         dtype=in_zarr.dtype,
                         chunks=in_zarr.chunks)

    reference = in_zarr[args.reference_index]
    reference -= reference.min()
    reference = np.floor(reference / reference.max() * args.nbins)
    reference_mask = reference > args.at

    indices = np.append(np.arange(args.reference_index),
                        np.arange(args.reference_index+1,
                                  in_zarr.shape[0]))

    # match all slices to reference and save all to output
    for i in tqdm(indices):
        target = in_zarr[i]
        matched_target = eqhist(reference, reference_mask,
                                target, args.at, args.nbins)
        out_zarr[i] = matched_target.astype(in_zarr.dtype) / float(args.nbins)
    out_zarr[args.reference_index] = reference.astype(in_zarr.dtype) / float(args.nbins)

    # fig, ax = plt.subplots(2, 3)
    # ax[0, 0].hist(reference.flatten(), bins=100)
    # ax[1, 0].imshow(reference, vmin=reference.min(), vmax=reference.max())
    # ax[0, 1].hist(target.flatten(), bins=100)
    # ax[1, 1].imshow(target, vmin=reference.min(), vmax=reference.max())
    # ax[0, 2].hist(matched_target.flatten(), bins=100)
    # ax[1, 2].imshow(matched_target, vmin=reference.min(), vmax=reference.max())
    # plt.show()


if __name__ == "__main__":
    main()
