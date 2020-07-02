import warnings
import numpy
import cupy
warnings.simplefilter(action='ignore', category=FutureWarning)


class MultivariateGaussianSum:
    """Allows efficient pdf lookup of multivarite Gaussian sum pdf for CPU or GPU code"""

    def __init__(self, means, covariances, weights, library=cupy):
        self.lib = library
        self.means = self.lib.asarray(means, dtype=self.lib.float32)
        self.weights = self.lib.asarray(weights, dtype=self.lib.float32)
        self.covariances = self.lib.asarray(covariances, dtype=self.lib.float32)

        self.inverse_covariances = self.lib.asarray(numpy.linalg.inv(covariances))
        self.Nd, self.Nx = means.shape

        self.constants = (2 * self.lib.pi) ** (-self.Nx / 2) / self.lib.sqrt(
            self.lib.linalg.det(self.covariances))

    def pdf(self, x):
        if len(x.shape) == 1:
            Np = 1
        else:
            Np = x.shape[0]
        x = x.reshape((Np, 1, self.Nx))
        means_device = self.means.reshape((1, self.Nd, self.Nx))
        es = x - means_device

        # The code below does: exp[i] = es[i].T @ self.inverse_covariances_device[i] @ es[i]
        exp = self.lib.einsum('...bc, ...c, ...b -> ...', self.inverse_covariances, es, es)
        r = self.lib.exp(-exp)

        # The code below does: result = sum(r[i] * self.weights_device[i] * self.constants_device[i])
        result = self.lib.einsum('...i, i, i -> ...', r, self.weights, self.constants)

        return result

    def draw(self, shape=(1,)):
        if not isinstance(shape, tuple):
            shape = (shape,)

        size = int(numpy.prod(shape))
        bins = self.lib.bincount(
            self.lib.random.choice(
                self.lib.arange(self.Nd),
                size,
                p=self.weights
            ),
            minlength=self.Nd
        )
        out = self.lib.empty((size, self.Nx), dtype=self.lib.float32)

        index = 0
        for n, mean, cov in zip(bins, self.means, self.covariances):
            out[index:index + n] = self.lib.random.multivariate_normal(mean, cov, int(n))
            index += n

        return out.reshape(shape + (self.Nx,)).squeeze()
