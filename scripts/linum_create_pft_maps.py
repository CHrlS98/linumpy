#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create inclusion and exclusion maps for PFT.
"""
import argparse
import nibabel as nib
import numpy as np

from linumpy.utils.io import assert_output_exists, add_overwrite_arg

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_wm_pve', help='Input WM partial volume estimation map.')
    p.add_argument('in_brain_mask',
                   help='Input brain mask (should include ventricles\n'
                        ' or else they will be added to include map).')
    p.add_argument('out_include', help='Output nifti image for inclusion map.')
    p.add_argument('out_exclude', help='Output nifti image for exclusion map.')
    p.add_argument('--endpoints_masks', nargs='+', required=True,
                   help='List of binary masks to use as endpoints for the inclusion map.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_exists(args.out_include, parser, args)
    assert_output_exists(args.out_exclude, parser, args)

    in_wm_pve = nib.load(args.in_wm_pve)
    wm_pve = in_wm_pve.get_fdata()

    # by default, all voxels belonging to gray matter/csf are excluded
    exclude = 1 - wm_pve

    # the voxels belonging to the inclusion map are all background
    # voxels + the voxels belonging to the endpoints masks
    include = np.zeros_like(wm_pve)
    for mask in args.endpoints_masks:
        mask = nib.load(mask).get_fdata() > 0
        include[mask] = exclude[mask]
        exclude[mask] = 0

    brain_mask = nib.load(args.in_brain_mask).get_fdata() > 0
    include[~brain_mask] = 1.0
    exclude[~brain_mask] = 0.0

    nib.save(nib.Nifti1Image(include.astype(np.float32), in_wm_pve.affine), args.out_include)
    nib.save(nib.Nifti1Image(exclude.astype(np.float32), in_wm_pve.affine), args.out_exclude)


if __name__ == '__main__':
    main()
