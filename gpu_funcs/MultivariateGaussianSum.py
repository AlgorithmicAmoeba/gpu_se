import numpy
import numba.cuda as cuda


class MultivariateGaussianSum:
    """Allows efficient pdf lookup of multivarite Gaussian sum pdf for GPU code"""

    def __init__(self, means, covariances, weights):
        self.means_device = cuda.to_device(means)
        self.weights_device = cuda.to_device(weights)

        self.inverse_covariances_device = cuda.to_device(numpy.linalg.inv(covariances))
        k = means.shape[1]

        constants = (2*numpy.pi)**(-k/2) / numpy.sqrt(numpy.linalg.det(covariances))
        self.constants_device = cuda.to_device(constants)

    @cuda.jit(device=True)
    def pdf(self, x):
        es = x - self.means_device
        result = 0
        for i in range(self.means_device.shape[1]):
            exp = es[i].T @ self.inverse_covariances_device[i] @ es[i]
            r = numpy.exp(exp)
            result += self.weights_device[i] * self.constants_device[i] * r

        return result
