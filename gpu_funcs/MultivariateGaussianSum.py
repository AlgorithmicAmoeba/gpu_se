import numpy
import numba.cuda as cuda
import cupy


class MultivariateGaussianSum:
    """Allows efficient pdf lookup of multivarite Gaussian sum pdf for GPU code"""

    def __init__(self, means, covariances, weights):
        self.means_device = cupy.asarray(means)
        self.weights_device = cupy.asarray(weights)
        self.covariances_device = cupy.asarray(covariances)

        self.inverse_covariances_device = cupy.asarray(numpy.linalg.inv(covariances))
        self.N, self.k = means.shape

        self.constants_device = (2*cupy.pi)**(-self.k/2) / cupy.sqrt(cupy.linalg.det(self.covariances_device))

    @cuda.jit(device=True)
    def pdf(self, x):
        es = x - self.means_device
        result = 0
        for i in range(self.means_device.shape[1]):
            exp = es[i].T @ self.inverse_covariances_device[i] @ es[i]
            r = numpy.exp(exp)
            result += self.weights_device[i] * self.constants_device[i] * r

        return result
