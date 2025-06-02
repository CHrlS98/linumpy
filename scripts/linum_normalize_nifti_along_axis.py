#!/usr/bin/env python3
"""
"""
import argparse
import numpy as np
import nibabel as nib


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image')
    p.add_argument('out_image')
    p.add_argument('x_bounds', type=int, nargs=2)
    p.add_argument('y_bounds', type=int, nargs=2)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    im = nib.load(args.in_image)
    data = im.get_fdata()

    xmin, xmax = args.x_bounds
    ymin, ymax = args.y_bounds
    roi = data[xmin:xmax, ymin:ymax]
    target = np.mean(roi)

    mean_per_slice = np.mean(roi, axis=(0, 1), keepdims=True)
    data /= mean_per_slice
    data *= target
    nib.save(nib.Nifti1Image(data, im.affine), args.out_image)


if __name__ == '__main__':
    main()
