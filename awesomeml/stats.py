# -*- coding: utf-8 -*-
"""
Statistics.
"""
import numpy as np
import scipy

def gaussian_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6): #pragma no cover
    """The Frechet distance between two Gaussians

    Code derived from:
    https://github.com/bioinf-jku/TTUR/blob/master/fid.py

    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Args:

      mu1 (ndarray): Mean vector of the first Gaussian

      sigma1 (ndarray): Covariance matrix of the first Gaussian

      mu2 (ndarray): Mean vector of the second Gaussian

      sigma2 (ndarray): Covariance matrix of the second Gaussian

    Returns:

    float: The Frechet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
