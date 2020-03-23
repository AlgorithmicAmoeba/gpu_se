import numpy
from filter.particle import ParallelParticleFilter
from gpu_funcs.MultivariateGaussianSum import MultivariateGaussianSum


def f(x, u, dt):
    x1, x2 = x

    dx1 = x1 - x2 * u
    dx2 = x1 / u + 2 * x2

    return x1 + dx1 * dt, x2 + dx2 * dt


def g(x, u):
    x1, x2 = x

    return [x1 * x2, x2 + u]


x0 = MultivariateGaussianSum(means=numpy.array([[10, 0],
                                                [-10, -10]]),
                             covariances=numpy.array([[[1, 0],
                                                       [0, 1]],

                                                      [[2, 0.5],
                                                       [0.5, 0.5]]]),
                             weights=numpy.array([0.3, 0.7]))

measurement_noise = MultivariateGaussianSum(means=numpy.array([[1, 0],
                                                              [0, -1]]),
                                            covariances=numpy.array([[[0.1, 0],
                                                                      [0, 0.1]],

                                                                     [[0.2, 0.01],
                                                                      [0.01, 0.005]]]),
                                            weights=numpy.array([0.85, 0.15]))

pp = ParallelParticleFilter(f, g, 10, x0, measurement_noise)

pp.predict(1, 1)
