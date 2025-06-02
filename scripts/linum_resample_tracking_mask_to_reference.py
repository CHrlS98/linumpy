#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resample high resolution tracking mask to match the reference of a FOD image. The
FOD image is expected to come from linum_orientation_structure_tensor.py
"""
import argparse
import numpy as np
import nibabel as nib


EPILOG="""
[1] Schilling et al, "Comparison of 3D orientation distribution functions
    measured with confocal microscopy and diffusion MRI". Neuroimage. 2016
    April 1; 129: 185-197. doi:10.1016/j.neuroimage.2016.01.022
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input image in .ome.zarr file format.')
    p.add_argument('out_image',
                   help='Output SH image (.nii.gz).')
    p.add_argument('reference',
                   help='Reference image.')
    
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    im = nib.load(args.in_image)
    source_voxel_size = im.header.get_zooms()[:3]
    data = im.get_fdata().astype(np.uint8)
    
    ref = nib.load(args.reference)
    new_voxel_size = ref.header.get_zooms()[:3]

    new_voxel_size_to_vox = (new_voxel_size / np.asarray(source_voxel_size)).astype(int)
    n_chunks_per_axis = np.ceil(np.asarray(data.shape) / np.asarray(new_voxel_size_to_vox)).astype(int)

    # 4. Create histogram for each new voxel
    out = np.zeros(n_chunks_per_axis)
    for chunk_x in range(n_chunks_per_axis[0]):
        for chunk_y in range(n_chunks_per_axis[1]):
            for chunk_z in range(n_chunks_per_axis[2]):
                content = data[chunk_x * new_voxel_size_to_vox[0]:
                               (chunk_x + 1) * new_voxel_size_to_vox[0],
                               chunk_y * new_voxel_size_to_vox[1]:
                               (chunk_y + 1) * new_voxel_size_to_vox[1],
                               chunk_z * new_voxel_size_to_vox[2]:
                               (chunk_z + 1) * new_voxel_size_to_vox[2]]
                out[chunk_x, chunk_y, chunk_z] = np.max(content)

    nib.save(nib.Nifti1Image(out.astype(np.uint8), ref.affine),
             args.out_image)


if __name__ == '__main__':
    main()
