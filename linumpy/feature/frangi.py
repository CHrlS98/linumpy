import numpy as np
from itertools import combinations_with_replacement
from warnings import warn

from skimage.feature import hessian_matrix
from skimage._shared.utils import check_nD, _supported_float_type


def _symmetric_compute_eigenvalues(S_elems):
    """Compute eigenvalues from the upper-diagonal entries of a symmetric
    matrix.

    Parameters
    ----------
    S_elems : list of ndarray
        The upper-diagonal elements of the matrix, as returned by
        `hessian_matrix` or `structure_tensor`.

    Returns
    -------
    eigs : ndarray
        The eigenvalues of the matrix, in decreasing order. The eigenvalues are
        the leading dimension. That is, ``eigs[i, j, k]`` contains the
        ith-largest eigenvalue at position (j, k).
    """

    if len(S_elems) == 3:  # Fast explicit formulas for 2D.
        M00, M01, M11 = S_elems
        eigs = np.empty((2, *M00.shape), M00.dtype)
        eigs[:] = (M00 + M11) / 2
        hsqrtdet = np.sqrt(M01**2 + ((M00 - M11) / 2) ** 2)
        eigs[0] += hsqrtdet
        eigs[1] -= hsqrtdet
        return eigs
    else:
        matrices = _symmetric_image(S_elems)
        # eigh returns eigenvalues in increasing order. We want decreasing
        eigs = np.linalg.eigvalsh(matrices)[..., ::-1]
        evals, evecs = np.linalg.eigh(matrices)
        evals = evals[..., ::-1]
        evecs = evecs[..., ::-1]

        leading_axes = tuple(range(eigs.ndim - 1))
        return np.transpose(eigs, (eigs.ndim - 1,) + leading_axes)


def _symmetric_image(S_elems):
    """Convert the upper-diagonal elements of a matrix to the full
    symmetric matrix.

    Parameters
    ----------
    S_elems : list of array
        The upper-diagonal elements of the matrix, as returned by
        `hessian_matrix` or `structure_tensor`.

    Returns
    -------
    image : array
        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,
        containing the matrix corresponding to each coordinate.
    """
    image = S_elems[0]
    symmetric_image = np.zeros(
        image.shape + (image.ndim, image.ndim), dtype=S_elems[0].dtype
    )
    for idx, (row, col) in enumerate(
        combinations_with_replacement(range(image.ndim), 2)
    ):
        symmetric_image[..., row, col] = S_elems[idx]
        symmetric_image[..., col, row] = S_elems[idx]
    return symmetric_image


def hessian_matrix_eigvals(H_elems):
    """Compute eigenvalues of Hessian matrix.

    Parameters
    ----------
    H_elems : list of ndarray
        The upper-diagonal elements of the Hessian matrix, as returned
        by `hessian_matrix`.

    Returns
    -------
    eigs : ndarray
        The eigenvalues of the Hessian matrix, in decreasing order. The
        eigenvalues are the leading dimension. That is, ``eigs[i, j, k]``
        contains the ith-largest eigenvalue at position (j, k).

    Examples
    --------
    >>> from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> H_elems = hessian_matrix(square, sigma=0.1, order='rc',
    ...                          use_gaussian_derivatives=False)
    >>> hessian_matrix_eigvals(H_elems)[0]
    array([[ 0.,  0.,  2.,  0.,  0.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 2.,  0., -2.,  0.,  2.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  2.,  0.,  0.]])
    """
    return _symmetric_compute_eigenvalues(H_elems)


def frangi_filter(
    image,
    sigmas=range(1, 10, 2),
    scale_range=None,
    scale_step=None,
    alpha=0.5,
    beta=0.5,
    gamma=None,
    black_ridges=True,
    mode='reflect',
    cval=0,
):
    """
    Filter an image with the Frangi vesselness filter.

    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvalues of the
    Hessian to compute the similarity of an image region to vessels, according
    to the method described in [1]_.

    Parameters
    ----------
    image : (M, N[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
        The default, None, uses half of the maximum Hessian norm.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (M, N[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Earlier versions of this filter were implemented by Marc Schrijver,
    (November 2001), D. J. Kroon, University of Twente (May 2009) [2]_, and
    D. G. Ellis (January 2017) [3]_.

    See also
    --------
    meijering
    sato
    hessian

    References
    ----------
    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi
    """
    if scale_range is not None and scale_step is not None:
        warn(
            'Use keyword parameter `sigmas` instead of `scale_range` and '
            '`scale_range` which will be removed in version 0.17.',
            stacklevel=2,
        )
        sigmas = np.arange(scale_range[0], scale_range[1], scale_step)

    check_nD(image, [2, 3])  # Check image dimensions.
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:  # Normalize to black ridges.
        image = -image

    # Generate empty array for storing maximum value
    # from different (sigma) scales
    filtered_max = np.zeros_like(image)
    filtered_dirs = np.zeros(image.shape + (3,), dtype=np.float32)

    for sigma in sigmas:  # Filter for all sigmas.
        H_elems = hessian_matrix(image, sigma, mode=mode, cval=cval,
                                 use_gaussian_derivatives=True)
        matrices = _symmetric_image(H_elems)
        eigvals, eigvecs = np.linalg.eigh(matrices)

        # move axes such that evals are along axis 0
        eigvals = np.moveaxis(eigvals, -1, 0)
        eigvecs = np.moveaxis(eigvecs, (-2, -1), (0, 1))

        # Sort eigenvalues by magnitude.
        eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)

        # This will be the principal direction if the filter response is maximal
        # for this scale
        eigvecs = np.take_along_axis(eigvecs, abs(eigvals).argsort(0)[None, ...], 1)[:, 0, ...]

        lambda1 = eigvals[0]
        if image.ndim == 2:
            (lambda2,) = np.maximum(eigvals[1:], 1e-10)
            r_a = np.inf  # implied by eq. (15).
            r_b = abs(lambda1) / lambda2  # eq. (15).
        else:  # ndim == 3
            lambda2, lambda3 = np.maximum(eigvals[1:], 1e-10)
            r_a = lambda2 / lambda3  # eq. (11).
            r_b = abs(lambda1) / np.sqrt(lambda2 * lambda3)  # eq. (10).
        s = np.sqrt((eigvals**2).sum(0))  # eq. (12).
        if gamma is None:
            gamma = s.max() / 2
            if gamma == 0:
                gamma = 1  # If s == 0 everywhere, gamma doesn't matter.
        # Filtered image, eq. (13) and (15).  Our implementation relies on the
        # blobness exponential factor underflowing to zero whenever the second
        # or third eigenvalues are negative (we clip them to 1e-10, to make r_b
        # very large).
        vals = 1.0 - np.exp(
            -(r_a**2) / (2 * alpha**2), dtype=image.dtype
        )  # plate sensitivity
        vals *= np.exp(-(r_b**2) / (2 * beta**2), dtype=image.dtype)  # blobness
        vals *= 1.0 - np.exp(
            -(s**2) / (2 * gamma**2), dtype=image.dtype
        )  # structuredness
        mask = vals > filtered_max
        filtered_max[mask] = vals[mask]
        filtered_dirs[mask] = np.moveaxis(eigvecs, 0, -1)[mask]

    return filtered_max, filtered_dirs  # Return pixel-wise max over all sigmas.
