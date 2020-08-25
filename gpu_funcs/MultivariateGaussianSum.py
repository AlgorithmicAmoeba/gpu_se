import warnings
import numpy
import cupy
warnings.simplefilter(action='ignore', category=FutureWarning)


class MultivariateGaussianSum:
    """Multivarite Gaussian distribution class for CPU and GPU implementations

    Parameters
    ----------
    means : library.array
        A (N_distributions x Nx) array of the mean for each Gaussian

    covariances : library.array
        A (N_distributions x Nx x Nx) array of the covariance for each Gaussian

    weights : library.array
        A (N_distributions) array of the weighting for each Gaussian

    library : {numpy, cupy}
        The library to be used for array operations.
        numpy is used for CPU implementations.
        cupy is used for GPU implementations
    """

    def __init__(self, means, covariances, weights, library=cupy):
        self.lib = library
        self.means = self.lib.asarray(means, dtype=self.lib.float32)
        self.weights = self.lib.asarray(weights, dtype=self.lib.float32)
        self.covariances = self.lib.asarray(covariances, dtype=self.lib.float32)

        self._inverse_covariances = self.lib.asarray(numpy.linalg.inv(covariances))
        self._Nd, self._Nx = means.shape

        self._constants = (2 * self.lib.pi) ** (-self._Nx / 2) / self.lib.sqrt(
            self.lib.linalg.det(self.covariances))

    def pdf(self, x):
        """Get the value of the probability density function evaluated at a point.

        Parameters
        ----------
        x : library.array
            A (m x Nx) array of points at which to evaluate the pdf

        Returns
        -------
        result : library.array
            A (m) array of pdf values

        """
        x = self.lib.atleast_2d(x)
        es = x[:, None, :] - self.means[None, :, :]

        # The code below does: exp[i] = es[i].T @ self.inverse_covariances_device[i] @ es[i]
        exp = es[:, :, None, :] @ self._inverse_covariances[None, :, :, :] @ es[:, :, :, None]
        r = self.lib.exp(-0.5*exp).squeeze()

        # The code below does: result = sum(r[i] * self.weights_device[i] * self.constants_device[i])
        result = numpy.sum(self.lib.atleast_2d(self._constants * self.weights * r), axis=1)

        return result

    def draw(self, shape=(1,)):
        """Draw samples from the distribution

        Parameters
        ----------
        shape : {int, tuple} (optional)
            Output shape

        Returns
        -------
        out : library.array
            A (*shape x Nx) array of samples
        """
        if not isinstance(shape, tuple):
            shape = (shape,)

        size = int(numpy.prod(shape))
        bins = self.lib.bincount(
            self.lib.random.choice(
                self.lib.arange(self._Nd),
                size,
                p=self.weights
            ),
            minlength=self._Nd
        )
        out = self.lib.empty((size, self._Nx), dtype=self.lib.float32)

        index = 0
        for n, mean, cov in zip(bins, self.means, self.covariances):
            out[index:index + n] = self.lib.random.multivariate_normal(mean, cov, int(n))
            index += n

        return out.reshape(shape + (self._Nx,))
