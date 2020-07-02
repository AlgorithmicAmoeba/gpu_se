import numpy
import cupy
import gpu_funcs.MultivariateGaussianSum


class DeterministicGaussianSum(gpu_funcs.MultivariateGaussianSum):
    """Creates a MultivariateGaussianSum object that always returns the same values.
    Useful for testing."""
    def __init__(self,  means, covariances, weights, library=cupy):
        super().__init__(means, covariances, weights, library=cupy)

        self.values = library.array([])

    def draw(self, shape=(1, )):
        if not isinstance(shape, tuple):
            shape = (shape,)

        size = int(numpy.prod(shape)) * self.Nx

        if self.values.size < size:
            drawn_vals = super().draw(size - self.values.size)
            self.values = self.lib.hstack([
                self.values,
                drawn_vals.flatten()
            ])

        out = self.values[:size]
        return out.reshape(shape + (self.Nx,)).squeeze()
