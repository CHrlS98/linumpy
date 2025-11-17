#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Plot distribution of out-of-plane angle from peaks image,
reference and slicing direction.
"""
import argparse
import nibabel as nib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_peaks', help='Input peaks.')
    p.add_argument('in_reference', help='Grayscale image for reference.')
    p.add_argument('out_plot', help='out_plot')
    p.add_argument('--slicing_direction', nargs=3, type=float,
                   default=(0.0, 0.0, 1.0),
                   help='Slicing acquisition direction. [%(default)s]')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Threshold applied to reference image to remove peaks\n'
                        'belonging to the background. [%(default)s]')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # slicing direction
    direction = np.array(args.slicing_direction).reshape((3, 1))
    direction /= np.linalg.norm(direction)

    im_peaks = nib.load(args.in_peaks)
    peaks = im_peaks.get_fdata()

    im_ref = nib.load(args.in_reference)
    reference = im_ref.get_fdata()
    mask = reference > args.threshold

    # reshape peaks
    peaks = np.squeeze(peaks)
    peaks = np.reshape(peaks, mask.shape + (3,))

    # take only peaks inside the mask AND with nonzero norm
    peaks = peaks[mask]
    peaks_norm = np.linalg.norm(peaks, axis=-1)
    peaks = peaks[peaks_norm > 0]
    peaks /= peaks_norm[peaks_norm > 0][:, None]

    phi = np.arccos(np.abs(peaks.dot(direction)))
    theta = np.pi / 2.0 - phi
    theta = np.rad2deg(theta)

    p95 = np.percentile(theta, 95)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 6)
    ax.hist(theta, bins=200)
    ax.axvline(p95)
    ax.set_xlabel('Out-of-plane angle (degrees)')
    ax.set_ylabel('Number of occurrences')
    ax.set_xticks(np.arange(0, 91, 10))
    fig.tight_layout()
    fig.savefig(args.out_plot)


if __name__ == '__main__':
    main()
