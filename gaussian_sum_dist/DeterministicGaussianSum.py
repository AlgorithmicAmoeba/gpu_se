import numpy
import cupy
import gaussian_sum_dist.MultivariateGaussianSum


class DeterministicGaussianSum(gaussian_sum_dist.MultivariateGaussianSum):
    """Creates a MultivariateGaussianSum singleton that always returns the same values.
    Useful for testing.

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

    __values = numpy.array([], dtype=numpy.float32)

    def __init__(self,  means, covariances, weights, library=cupy):
        super().__init__(means, covariances, weights, library)

    def draw(self, shape=(1, )):
        """Draw samples from the distribution

        Parameters
        ----------
        shape : {int, tuple} (optional)
            Output shape

        Returns
        -------
        out : library.array
            A (\*shape x Nx) array of samples
        """
        if not isinstance(shape, tuple):
            shape = (shape,)

        size = int(numpy.prod(shape)) * self._Nx

        if DeterministicGaussianSum.__values.size < size:
            drawn_vals = super().draw(size - DeterministicGaussianSum.__values.size)

            if self.lib == cupy:
                drawn_vals = drawn_vals.get()

            DeterministicGaussianSum.__values = numpy.hstack([
                DeterministicGaussianSum.__values,
                drawn_vals.flatten()
            ])

        out = DeterministicGaussianSum.__values[:size]
        out = out.reshape(shape + (self._Nx,)).squeeze()
        if self.lib == cupy:
            out = cupy.asarray(out)
        return out
