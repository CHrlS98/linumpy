#!/usr/bin/env python3
import argparse
import zarr
import numpy as np

from skimage.registration import phase_cross_correlation
from skimage.transform import pyramid_gaussian
from scipy.ndimage import center_of_mass, shift
from tqdm import tqdm
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image')
    p.add_argument('out_image')
    p.add_argument('--intermediate_output')
    p.add_argument('--reference_index', type=int)
    p.add_argument('--align_center_of_mass', action='store_true')
    p.add_argument('--pyramid_level', type=int, default=4)
    return p


def register(fixed, moving, pyramid_level):
    fixed_sub = tuple(pyramid_gaussian(fixed, max_layer=pyramid_level))[-1]
    moving_sub = tuple(pyramid_gaussian(moving, max_layer=pyramid_level))[-1]
    px_shift, _, _ = phase_cross_correlation(fixed_sub, moving_sub)
    return shift(moving, px_shift*2**pyramid_level)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    im = zarr.open(args.in_image, mode='r')
    reference_index = args.reference_index
    if reference_index is None:
        reference_index = im.shape[0] // 2

    out = zarr.open(args.out_image, mode='w',
                    shape=im.shape, chunks=im.chunks)
    if args.align_center_of_mass:
        (rmid, cmid) = im.shape[1] / 2, im.shape[2] / 2
        for z in tqdm(range(im.shape[0])):
            rmass, cmass = center_of_mass(im[z])
            out[z] = shift(im[z], (rmid - rmass, cmid - cmass))
    else:
        out[:] = im

    if args.intermediate_output:
        intermediate_output = zarr.open(args.intermediate_output, mode='w',
                                        shape=im.shape, chunks=im.chunks)
        intermediate_output[:] = out

    new_reference = reference_index
    for z in tqdm(range(reference_index+1, im.shape[0]), 'increasing index'):
        out[z] = register(out[new_reference], out[z], args.pyramid_level)
        new_reference = z

    new_reference = reference_index
    for z in tqdm(range(reference_index - 1, -1, -1), 'decreasing index'):
        out[z] = register(out[new_reference], out[z], args.pyramid_level)
        new_reference = z


if __name__ == '__main__':
    main()
