#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import argparse
import re
from pathlib import Path
import numpy as np
from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.stitching.registration import apply_transform
from tqdm import tqdm

import dask.array as da
import zarr
import SimpleITK as sitk


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_stack',
                   help='Input stack in .ome.zarr format.')
    p.add_argument('in_offsets',
                   help='Input offsets file (.npy) with z offsets for each slice.')
    p.add_argument('in_transforms_dir',
                   help='Directory containing transformations (.mat) between consecutive slices.\n' \
                        'Expected filename format is "*z{moving_slice_id}_*".')
    p.add_argument('out_stack',
                   help='Output stack in .ome.zarr format.')
    p.add_argument("--slicing_interval", type=float, default=0.200,
                   help="Interval between slices in mm. [%(default)s]")
    p.add_argument("--factor_extra", type=float, default=1.1,
                   help='Factor by which to increase the stacking interval. [%(default)s]')
    p.add_argument('--ignore', nargs='+', type=int,
                   help="Slice indices to ignore in the transformation application.")
    p.add_argument('--first_slice_index', type=int, default=0,
                   help='Index of the first slice in the stack. [%(default)s]')
    p.add_argument('--keep_extra', action='store_true',
                   help='Keep extra slices (useful for performing additional registration).')
    return p


def apply_transform_3d(idx, in_vol, out_vol, all_transforms,
                       offset_source, offset_dest):
    num_voxels_dest = offset_dest[idx + 1] - offset_dest[idx]
    curr_vol = in_vol[offset_source[idx]:offset_source[idx] + num_voxels_dest]
    transforms = all_transforms[idx - 1::-1]
    composite_transform = sitk.CompositeTransform(transforms)

    out = apply_transform(curr_vol, composite_transform)
    out_vol[offset_dest[idx]:offset_dest[idx+1]] = out


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_stack)
    interval_vox = int(np.ceil(args.slicing_interval / res[0]))  # in voxels

    z_offsets_source = np.load(args.in_offsets)
    n_slices = len(z_offsets_source)
    # for the final slice of the volume
    z_offsets_source = np.append(z_offsets_source, vol.shape[0])
    if args.keep_extra:
        z_offsets_dest = z_offsets_source
        n_overlap_vox = 0
    else:
        n_overlap_vox = int(interval_vox * (args.factor_extra - 1.0))  # in voxels
        overlap_to_remove = np.arange(len(z_offsets_source)) * n_overlap_vox
        z_offsets_dest = z_offsets_source - overlap_to_remove
        z_offsets_dest = np.append(z_offsets_dest, z_offsets_dest[-1] + interval_vox)
    print(n_slices, "slices in the source stack.")
    
    transforms = [sitk.Transform()] * (n_slices)
    transforms_files = Path(args.in_transforms_dir).glob("*.mat")
    pattern = r".*z(\d+).*"  # the parentheses create a group containing the slice id
    for f in transforms_files:
        match = re.match(pattern, f.name)
        # the only group found is the slice id
        id = int(match.groups()[0])
        transforms[id - args.first_slice_index] = sitk.ReadTransform(f)

    if None in transforms[1:]:
        raise ValueError("Not all transforms were found. "
                         "Check the input directory and the file naming convention.")

    # skip the first slice since it has no transform
    transforms[0].SetIdentity()

    # optionally ignore some slices
    if args.ignore is not None:
        for i in args.ignore:
            transforms[i - args.first_slice_index].SetIdentity()

    _, nr, nc = vol.shape

    output_shape = (z_offsets_dest[-1], nr, nc)
    output_vol = zarr.open(zarr.TempStore(), mode='w', shape=output_shape,
                           chunks=vol.chunks, dtype=vol.dtype)
    output_vol[z_offsets_dest[0]:z_offsets_dest[1] + n_overlap_vox] =\
        vol[z_offsets_source[0]:z_offsets_source[1]]  # we keep the overlap region in case keep_extra is True

    # assemble volume
    for i in tqdm(range(1, n_slices), desc='Apply transforms to volume'):
        curr_vol = vol[z_offsets_source[i]:z_offsets_source[i + 1]]
        composite_transform = sitk.CompositeTransform(transforms[i::-1])
        reg_vol = apply_transform(curr_vol, composite_transform)

        z_max = min(z_offsets_dest[i + 1] + n_overlap_vox, output_vol.shape[0])
        output_vol[z_offsets_dest[i]:z_max] = reg_vol[:z_max-z_offsets_dest[i]]

    save_omezarr(da.from_zarr(output_vol), args.out_stack,
                 voxel_size=res, chunks=vol.chunks)


if __name__ == "__main__":
    main()
