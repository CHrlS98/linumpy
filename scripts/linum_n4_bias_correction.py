#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import argparse
import numpy as np
import SimpleITK as sitk
from linumpy.io.zarr import read_omezarr, save_omezarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dask.array as da


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image')
    p.add_argument('out_image')

    p.add_argument('out_screenshot')
    return p


def get_cross_section(volume):
    return volume[:, volume.shape[1], :]


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol, res = read_omezarr(args.in_image, level=1)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    inputImage = sitk.GetImageFromArray(vol[:])
    image = inputImage

    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    numberFittingLevels = 4

    corrector.SetMaximumNumberOfIterations([50]*numberFittingLevels)

    corrected_image = corrector.Execute(image, maskImage)

    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)

    corrected_vol = sitk.GetArrayFromImage(corrected_image_full_resolution)
    save_omezarr(da.from_array(corrected_vol), args.out_image, voxel_size=res)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].imshow(get_cross_section(vol))
    axes[1].imshow(get_cross_section(corrected_vol))
    fig.tight_layout()
    fig.savefig(args.out_screenshot)


if __name__ == '__main__':
    main()
