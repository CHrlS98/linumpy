#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    d = np.array([1.0, 1.0, 1.0])
    d /= np.linalg.norm(d)

    print("input direction is: ", d)

    n_sagittal = np.array([1, 0, 0])
    n_coronal = np.array([0, 1, 0])
    n_axial = np.array([0, 0, 1])

    sagittal = np.sin(np.arccos(d.dot(n_sagittal.T)))
    coronal = np.sin(np.arccos(d.dot(n_coronal.T)))
    axial = np.sin(np.arccos(d.dot(n_axial.T)))

    I = np.array([sagittal, coronal, axial])

    print("intensities are: ", I)

    A = np.array([n_sagittal,
                  n_coronal,
                  n_axial])
    B = np.cos(np.arcsin(I))
    print(A, B)

    x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(B)
    print(x)


if __name__ == '__main__':
    main()
