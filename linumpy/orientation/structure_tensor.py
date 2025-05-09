# -*- coding: utf-8 -*-
from linumpy.orientation.gradient import riesz
from scipy.ndimage import gaussian_filter
import numpy as np


def structure_tensor(I, sigma, method='riesz',
                     prefilter_sigma=None, return_gradient=False):
    """
    Compute the structure tensor of a nD image `I`.
    """
    if prefilter_sigma is not None:
        I = gaussian_filter(I, prefilter_sigma, mode='constant')
    if method == 'riesz':
        derivatives = riesz(I)
    elif method == 'gradient':
        derivatives = np.gradient(I)
    else:
        raise ValueError("Method must be 'riesz' or 'gradient'.")
    # Compute the structure tensor
    ST = np.zeros(np.append(I.shape, [len(derivatives), len(derivatives)]), dtype=np.float64)
    for i in range(len(derivatives)):
        for j in range(i, len(derivatives)):
            ST[..., i, j] = gaussian_filter(derivatives[i]*derivatives[j], sigma=sigma, mode='reflect')
            ST[..., j, i] = ST[..., i, j]

    if return_gradient:
        return ST, derivatives
    return ST
