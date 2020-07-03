import numpy
import cupy
import gpu_funcs.MultivariateGaussianSum


class DeterministicGaussianSum(gpu_funcs.MultivariateGaussianSum):
    """Creates a MultivariateGaussianSum singleton that always returns the same values.
    Useful for testing."""

    values = numpy.array([], dtype=numpy.float32)

    def __init__(self,  means, covariances, weights, library=cupy):
        super().__init__(means, covariances, weights, library)

    def draw(self, shape=(1, )):
        if not isinstance(shape, tuple):
            shape = (shape,)

        size = int(numpy.prod(shape)) * self.Nx

        if DeterministicGaussianSum.values.size < size:
            drawn_vals = super().draw(size - DeterministicGaussianSum.values.size)

            if self.lib == cupy:
                drawn_vals = drawn_vals.get()

            DeterministicGaussianSum.values = numpy.hstack([
                DeterministicGaussianSum.values,
                drawn_vals.flatten()
            ])

        out = DeterministicGaussianSum.values[:size]
        out = out.reshape(shape + (self.Nx,)).squeeze()
        if self.lib == cupy:
            out = cupy.asarray(out)
        return out
