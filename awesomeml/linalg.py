# -*- coding: utf-8 -*-
"""
Linear algebra functionalities.
"""
import itertools
import scipy
import scipy.linalg
import numpy as np
from . import utils as _utils

def gram_schmidt(X, axis1=0, axis2=None, normalize=True):
    """Perform Gram-Schmidt procedure on a matrix (or matrices)

    Args:

      X (ndarray): Input matrix (or matrices).

      axis1 (int, Seq[int]): Axis (or axes) to define the first
        dimension of the matrix. The whole dimension will be
        considered as a column vector which will be normalized. If
        None, all the axes that are not in axis2 will be used. Default
        to 0.

      axis2 (int, Seq[int]): Axis (or axes) to define the second
        dimension of the matrix. Along this dimension, vectors will be
        orthogonalized. If None, all the axes that are not in axis1
        will be used. Default to None.

      normalize (bool): Whether normalize vectors defined by axis1 so
        that they have l2-norm with value 1. Default to true.

    Examples:

    >>> X = np.random.random((3,))
    >>> Y = gram_schmidt(X)
    >>> np.testing.assert_almost_equal(np.sum(np.square(Y)), 1)
    >>> X = np.random.random((3, 3))
    >>> Y = gram_schmidt(X)
    >>> np.testing.assert_almost_equal(Y.T @ Y, np.eye(3))
    >>> X = np.random.random((3, 3, 6))
    >>> Y = gram_schmidt(X, axis1=(0, 1))
    >>> Y = np.reshape(Y, [9, 6])
    >>> np.testing.assert_almost_equal(Y.T @ Y, np.eye(6))
    >>> X = np.random.random((3, 3, 3, 3))
    >>> Y = gram_schmidt(X, axis1=None, axis2=(2, 3))
    >>> Y = np.reshape(Y, [9, 9])
    >>> np.testing.assert_almost_equal(Y.T @ Y, np.eye(9))
    >>> X = np.random.random((3, 3, 3, 3))
    >>> Y = gram_schmidt(X, axis1=0, axis2=2)
    >>> for i in range(3):
    ...     for j in range(3):
    ...         Y2 = Y[:, i, :, j]
    ...         np.testing.assert_almost_equal(Y2.T @ Y2, np.eye(3))
    >>> gram_schmidt(X, axis1=None, axis2=None)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> gram_schmidt(X, [1, 2], [0, 1])
    Traceback (most recent call last):
      ...
    ValueError: ...

    """
    ndim = X.ndim
    axis1 = _utils.validate_axis(axis1, ndim, 'axis1',
                                 accept_none=True,
                                 scalar_to_seq=True)
    axis2 = _utils.validate_axis(axis2, ndim, 'axis2',
                                 accept_none=True,
                                 scalar_to_seq=True)
    
    if axis1 is None and axis2 is None:
        raise ValueError('Both axis1 and axis2 are None')
    if axis1 is None:
        axis1 = list(set(range(ndim)) - set(axis2))
    if axis2 is None:
        axis2 = list(set(range(ndim)) - set(axis1))
    if set(axis1) & set(axis2):
        raise ValueError('axis1 and axis2 must not contain any same value.')

    def dot(u, v):
        return np.sum(u * v, tuple(axis1), keepdims=True)

    def proj(u, v):
        return u * dot(v, u) / dot(u, u)

    U = np.copy(X)
    indices = [range(X.shape[i]) if i in axis2 else [None]
               for i in range(ndim)]
    for I in itertools.product(*indices):
        I = tuple(slice(None) if i is None else slice(i, i+1) for i in I)
        for J in itertools.product(*indices):
            J = tuple(slice(None) if j is None else slice(j, j+1) for j in J)
            if J == I:
                break
            U[I] -= proj(U[J], X[I])

    # normalization
    if normalize:
        U /= np.sqrt(np.sum(np.square(U), tuple(axis1), keepdims=True))
    return U
