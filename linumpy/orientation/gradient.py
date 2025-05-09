# -*- coding: utf-8 -*-
from scipy import fft
import numpy as np


def riesz(f):
    F = fft.fftn(f)
    freqs = [fft.fftfreq(d) for d in F.shape]
    freqs_grids = np.meshgrid(*freqs, indexing='ij')
    freqnorms = np.linalg.norm(np.stack(freqs_grids, axis=-1), axis=-1)

    transforms = []
    for freqs in freqs_grids:
        H = 1.0j * freqs
        H[freqnorms > 0.0] /= freqnorms[freqnorms > 0.0]
        fh = fft.ifftn(F*H).real  # input is real, so is output
        transforms.append(fh)

    return tuple(transforms)
