import numpy
import gaussian_sum_dist.DeterministicGaussianSum


def test_DeterministicGaussianSum():
    m = gaussian_sum_dist.DeterministicGaussianSum(
        means=numpy.array([[10, 0],
                           [-10, -10]]),
        covariances=numpy.array([[[1, 0],
                                  [0, 1]],

                                 [[2, 0.5],
                                  [0.5, 0.5]]]),
        weights=numpy.array([0.3, 0.7]))

    draw1 = m.draw(60)
    draw2 = m.draw((10, 7, 6))
    assert numpy.sum(draw1 - m.draw(60)) == 0.
    assert numpy.sum(draw2 - m.draw((10, 7, 6))) == 0.
