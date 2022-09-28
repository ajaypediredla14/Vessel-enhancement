from warnings import warn

import numpy as np

from .._shared.utils import _supported_float_type, check_nD
from ..feature.corner import hessian_matrix, hessian_matrix_eigvals
from ..util import img_as_float, invert


def sato(image, sigmas=range(1, 10, 2), black_ridges=True,
         mode='reflect', cval=0):
    """
    Filter an image with the Sato tubeness filter.

    This filter can be used to detect continuous ridges, e.g. tubes,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvectors of the
    Hessian to compute the similarity of an image region to tubes, according to
    the method described in [1]_.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter.
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
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    See also
    --------
    meijering
    frangi
    hessian

    References
    ----------
    .. [1] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
        Koller, T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line
        filter for segmentation and visualization of curvilinear structures in
        medical images. Medical image analysis, 2(2), 143-168.
        :DOI:`10.1016/S1361-8415(98)80009-1`
    """

    # Check image dimensions
    check_nD(image, [2, 3])

    # Check (sigma) scales
    sigmas = _check_sigmas(sigmas)

    # Invert image to detect bright ridges on dark background
    if not black_ridges:
        image = invert(image)

    float_dtype = _supported_float_type(image.dtype)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered
    # at different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape, dtype=float_dtype)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (sorted) eigenvalues
        lamba1, *lambdas = compute_hessian_eigenvalues(image, sigma,
                                                       sorting='val',
                                                       mode=mode, cval=cval)

        # Compute tubeness, see  equation (9) in reference [1]_.
        # np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
        filtered = np.abs(np.multiply.reduce(lambdas)) ** (1/len(lambdas))

        # Remove background and store results in (n+1)D matrices
        filtered_array[i] = np.where(lambdas[-1] > 0, filtered, 0)

    # Return for every pixel the maximum value over all (sigma) scales
    return np.max(filtered_array, axis=0)
