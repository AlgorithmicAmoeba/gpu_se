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

    def pdf(self, x):
        x = cupy.asarray(x)
        
        es = x - self.means_device
        # exp[i] = es[i].T @ self.inverse_covariances_device[i] @ es[i]
        exp = cupy.einsum('abc, cd, ed -> a', self.inverse_covariances_device, es, es)
        r = cupy.exp(-exp)
        # result = sum(r[i] * self.weights_device[i] * self.constants_device[i])
        result = cupy.einsum('i, i, i -> ', r, self.weights_device, self.constants_device)

        return result

    def draw(self, shape=(1,)):
        if isinstance(shape, int):
            shape = (shape, )

        size = int(numpy.prod(shape))
        bins = cupy.bincount(cupy.random.choice(cupy.arange(self.N), size, p=self.weights_device), minlength=self.N)
        out = cupy.empty((size, self.k))

        index = 0
        for n, mean, cov in zip(bins, self.means_device, self.covariances_device):
            out[index:index+n] = cupy.random.multivariate_normal(mean, cov, int(n))
            index += n

        return out.reshape(shape + (self.k, ))
