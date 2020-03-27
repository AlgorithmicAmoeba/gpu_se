import numpy
import numba.cuda as cuda
import cupy


class MultivariateGaussianSum:
    """Allows efficient pdf lookup of multivarite Gaussian sum pdf for CPU or GPU code"""

    def __init__(self, means, covariances, weights, library=cupy):
        self.lib = library
        self.means_device = self.lib.asarray(means)
        self.weights_device = self.lib.asarray(weights)
        self.covariances_device = self.lib.asarray(covariances)

        self.inverse_covariances_device = self.lib.asarray(numpy.linalg.inv(covariances))
        self.Nd, self.Nx = means.shape

        self.constants_device = (2 * self.lib.pi) ** (-self.Nx / 2) / self.lib.sqrt(
            self.lib.linalg.det(self.covariances_device))

    def pdf(self, x):
        if len(x.shape) == 1:
            Np = 1
        else:
            Np = x.shape[0]
        x = x.reshape((Np, 1, self.Nx))
        means_device = self.means_device.reshape((1, self.Nd, self.Nx))
        es = x - means_device

        # The code below does: exp[i] = es[i].T @ self.inverse_covariances_device[i] @ es[i]
        exp = self.lib.einsum('...bc, ...c, ...b -> ...', self.inverse_covariances_device, es, es)
        r = self.lib.exp(-exp)

        # The code below does: result = sum(r[i] * self.weights_device[i] * self.constants_device[i])
        result = self.lib.einsum('...i, i, i -> ...', r, self.weights_device, self.constants_device)

        return result

    def draw(self, shape=(1,)):
        if isinstance(shape, int):
            shape = (shape,)

        size = int(numpy.prod(shape))
        bins = self.lib.bincount(self.lib.random.choice(self.lib.arange(self.Nd), size, p=self.weights_device),
                                 minlength=self.Nd)
        out = self.lib.empty((size, self.Nx))

        index = 0
        for n, mean, cov in zip(bins, self.means_device, self.covariances_device):
            out[index:index + n] = self.lib.random.multivariate_normal(mean, cov, int(n))
            index += n

        return out.reshape(shape + (self.Nx,))
