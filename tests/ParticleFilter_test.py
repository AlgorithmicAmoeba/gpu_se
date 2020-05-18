import numpy
from filter.particle import ParticleFilter
from gpu_funcs.MultivariateGaussianSum import MultivariateGaussianSum


def f(x, u, dt):
    x1, x2 = x

    dx1 = x1 - x2 * u
    dx2 = x1 / u + 2 * x2

    return x1 + dx1 * dt, x2 + dx2 * dt


def g(x, u):
    x1, x2 = x

    return x1 * x2, x2 + u


x0 = MultivariateGaussianSum(
    means=numpy.array([[10, 0],
                       [-10, -10]]),
    covariances=numpy.array([[[1, 0],
                              [0, 1]],
                             [[2, 0.5],
                              [0.5, 0.5]]]),
    weights=numpy.array([0.3, 0.7]),
    library=numpy)

state_noise = MultivariateGaussianSum(
    means=numpy.array([[1e-3, 0],
                       [0, -1e-3]]),
    covariances=numpy.array([[[1e-4, 0],
                              [0, 1e-5]],
                             [[2e-4, 1e-5],
                              [1e-5, 5e-6]]]),
    weights=numpy.array([0.5, 0.5]),
    library=numpy
)

measurement_noise = MultivariateGaussianSum(
    means=numpy.array([[1, 0],
                       [0, -1]]),
    covariances=numpy.array([[[0.1, 0],
                              [0, 0.1]],
                             [[0.2, 0.01],
                              [0.01, 0.005]]]),
    weights=numpy.array([0.7, 0.9]),
    library=numpy
)

p = ParticleFilter(f, g, 10, x0, state_noise, measurement_noise)


def test_ParticleFilter_predict():
    p.predict(1, 1)


def test_ParticleFilter_update():
    z = numpy.array([2.3, 1.2])
    p.update(1.0, z)


def test_ParticleFilter_resample():
    p.resample()
