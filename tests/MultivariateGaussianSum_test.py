import numpy
import cupy
from gpu_funcs.MultivariateGaussianSum import MultivariateGaussianSum

means = numpy.array([[10, 0],
                     [-10, -10]])
covariances = numpy.array([[[1, 0],
                            [0, 1]],

                           [[2, 0.5],
                            [0.5, 0.5]]])

weights = numpy.array([0.3, 0.7])

m = MultivariateGaussianSum(means, covariances, weights)

x = cupy.array([-10, -10])
pdf_test = m.pdf(x)

draw_test = m.draw(10)
