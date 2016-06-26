# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .kern import Kern
from ...core import Param
from paramz.transformations import Logexp

class SpectralMixture(Kern):
    """
    Spectral Mixture Kernel (covariance function).

    This kernel models the spectral density (Fourier transform)
    of a kernel as a mixture of Gaussians with diagonal covariance,
    as introduced by Andrew Wilson and Ryan Adams in

        http://hips.seas.harvard.edu/files/wilson-extrapolation-icml-2013_0.pdf
        http://www.cs.cmu.edu/~andrewgw/typo.pdf

    The covariance is of the form

    .. math::

        k(\\tau) = \sum_q w_q \\cos(2\\pi \\tau^T\\mu_q) \\prod_p \\exp(-2\\pi^2 \\tau^2 v_q^{(p)})

    where

        \\tau = x - x',
        \\mu_q  is the mean of the q-th frequency space Gaussian
        w_q     is the weight of the q-th Gaussian
        v_q   = diag(\\Sigma_q) is the diagonal covariance matrix
        p       is each dimension in the input space

    By definition, k is stationary.

    :param q: the number of Gaussians in the spectral mixture
    :type  q: int
    :param w: a [q x 1] matrix of weights for each SM kernel in the sum
    :type  w: array-like
    :param variances: a [p x q] matrix where each column log_variance[:,q] = sigma_q^2 = v_q
    :type  variances: array-like
    :param means: a [p x q] matrix where each column mean[:,q] = \\mu_q
    :type  means: array-like
    :rtype: kernel object
    """

    def __init__(self, q, w, means=None, variances=None, input_dim=1,
                 active_dims=None, name='SpectralMixture'):
        super(SpectralMixture, self).__init__(input_dim, active_dims, name)

        assert q > 0, 'Number of Gaussians must be non-negative'

        if w is None:
            w = np.random.rand(q,1)
        else:
            # check that dimensions are correct
            assert w.shape == (q,1), 'Weights must be [q x 1]'

        if variances is None:
            variances = np.ones((p, q))
        else:
            # check that the dimensions are correct
            assert variance.shape == (p,q), 'Variance matrix must be [p x q]'

        if means is None:
            # randomly instantiate the means
            means = np.random.randn(p, q)
        else:
            # check that the dimensions are correct
            assert means.shape == (p,q), 'Means matrix must be [p x q]'

        self.q = q
        self.w = Param('weights', w, Logexp())
        self.means = Param('means', means)
        self.variances = Param('variances', variances, Logexp())

    def K(self, X, X2=None):
        """
        K defines the covariance matrix between input
        matrices X and X2 such that

        .. math::
            K_{ij} = k(X_i, X2_j)

        if X2 is None then we treat it as if X2 == X.

        :param X: a [n x p] matrix input
        :type  X: array-like
        :param X2: a [m x p] matrix input
        :type  X2: array-like
        :rtype: an [n x m] np.ndarray
        """
        if X2 is None:
            X2 = X
        
        # define tau[i,j] = X[i] - X2[j] \in \RR^{X.shape[1]}
        # so tau.shape == (m,n,p)
        tau = X[:,np.newaxis,:] - X2
        K = np.zeros_like(tau)

        # tau(m,n,p) tensordot means(p,q) -> dot_prod(m,n,q)
        # where dot_prod[i,j,k] = tau[i,j]'*means[:,k]
        K = np.cos(2*np.pi*np.tensordot(tau, mu, axes=1)) *\
            np.exp(-2 * np.pi**2 * np.tensordot(tau**2, v, axes=1))

        # return the weighted sum of the individual
        # Gaussian kernels, dropping the third index
        return np.tensordot(K, w, axis=1).squeeze(axis=(2,))

    def Kdiag(self, X):
        """
        Kdiag defines the diagonal of the kernel matrix
        between X and itself such that

        .. math::
            Kdiag_i = k(X_i, X_i)

        Note that we know

        .. math::
            \\tau = 0,

        so the kernel simplifies to 

        .. math::
            k(x_i, x_i) = \sum_q w_q

        :param X: a [n x p] matrix input
        :type  X: array-like
        :rtype: an [n x 1] np.ndarray
        """
        return self.w.sum() * np.ones((X.shape[0]))

    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        update_gradients_full takes in the derivative of the marginal
        likelihood with respect to the kernel, and one or two input
        matrices, updating the state of the gradients of the marginal
        likelihood with respect to each parameter of the model.

        :param dL_dK: the derivative of the marginal likelihood with
                      respect to the kernel output
        """
        pass
