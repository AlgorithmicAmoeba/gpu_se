import numpy.random


class WhiteGaussianNoise:
    """Contains information on the random variable used to generate white gaussian noise
    """
    def __init__(self, covariance):
        self.covariance = covariance
        N = self.covariance.shape[0]
        self.mean = numpy.zeros(N)

    def __call__(self):
        return numpy.random.multivariate_normal(self.mean, self.covariance)

