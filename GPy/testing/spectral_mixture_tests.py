# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest

import numpy as np

import GPy.kern

class SpectralMixtureKernelTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_one_gaussian_one_dim(self):
        q = 1
        w = np.array([1.5]).reshape(-1,1)
        means = np.array([0.5]).reshape(-1,1)
        variances = np.array([1.0]).reshape(-1,1)
        
        k = GPy.kern.SpectralMixture(q=q, w=w, means=means, variances=variances)
        X = np.linspace(-10, 10, 11).reshape(-1,1)
        X2 = np.linspace(-10, 10, 23).reshape(-1,1)
        
        np.testing.assert_equal(k.Kdiag(X), float(w))

        K = k.K(X, X2)
        assert K.shape == (X.shape[0], X2.shape[0])

        for i, x in enumerate(X):
            for j, y in enumerate(X2):
                x = np.array(x).reshape(1,1)
                y = np.array(y).reshape(1,1)

                np.testing.assert_almost_equal(
                    k.K(x, y),
                    K[i,j],
                    err_msg='K(X[i],X2[j]) != K[i,j]',
                )

                np.testing.assert_almost_equal(
                    k.K(x, y),
                    w * np.exp(
                        -2 * np.pi**2 * (x - y)**2 * variances
                    ) * np.cos(
                        2 * np.pi * (x-y) * means
                    ),
                    err_msg='K(x,y) != actual formula',
                )

    def test_two_gaussian_two_dim(self):
        q = 2
        w = np.array([1.5, 0.75]).reshape(-1,1)
        means = np.array([[-0.3, -0.2], [0.5, 10.2]])
        variances = np.array([[0.2, 4.2], [1.0, 0.01]])
        
        k = GPy.kern.SpectralMixture(q=q, w=w, means=means, variances=variances, input_dim=2)
        X = np.hstack((
            np.linspace(-10, 10, 11).reshape(-1,1),
            np.linspace(5, 10, 11).reshape(-1,1),
        ))
        X2 = np.hstack((
            np.linspace(-10, 10, 23).reshape(-1,1),
            np.linspace(-3, 5, 23).reshape(-1,1),
        ))
        
        np.testing.assert_equal(k.Kdiag(X), w.sum())

        K = k.K(X, X2)
        assert K.shape == (X.shape[0], X2.shape[0])

        for i, x in enumerate(X):
            assert k.K(x.reshape(1,-1)) == w.sum()

            for j, y in enumerate(X2):
                x = np.array(x).reshape(1,-1)
                y = np.array(y).reshape(1,-1)
                print(i, j, x, y)

                np.testing.assert_almost_equal(
                    k.K(x, y),
                    K[i,j],
                    err_msg='K(X[i],X2[j]) != K[i,j]',
                )

                should = w[0] * np.exp(
                    -2 * np.pi**2 * (x[0,0] - y[0,0])**2 * variances[0,0]
                    -2 * np.pi**2 * (x[0,1] - y[0,1])**2 * variances[1,0]
                ) * np.cos(
                    2 * np.pi * (x - y) @ means[:,0].reshape(-1,1)
                ) + w[1] * np.exp(
                    -2 * np.pi**2 * (x[0,0] - y[0,0])**2 * variances[0,1]
                    -2 * np.pi**2 * (x[0,1] - y[0,1])**2 * variances[1,1]
                ) * np.cos(
                    2 * np.pi * (x - y) @ means[:,1].reshape(-1,1)
                )

                np.testing.assert_almost_equal(
                    k.K(x, y),
                    should,
                    err_msg='K(x,y) != actual formula',
                )


if __name__ == '__main__':
    unittest.main()
