import numpy.random


class Noise:
    """Base class for noise generators"""
    def __call__(self):
        raise NotImplementedError

    def cov(self):
        raise NotImplementedError


class WhiteGaussianNoise(Noise):
    """Contains information on the random variable used to generate white gaussian noise
    """
    def __init__(self, covariance):
        self._covariance = covariance
        N = self._covariance.shape[0]
        self.mean = numpy.zeros(N)

    def __call__(self):
        return numpy.random.multivariate_normal(self.mean, self._covariance)

    def cov(self):
        return self._covariance

