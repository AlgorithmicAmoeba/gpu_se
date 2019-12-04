import numpy
import noise
import model
import controller

# Noise
state_noise_cov = numpy.array([[1e-1, 0], [0, 7e-2]])
meas_noise_cov = numpy.array([[6e-2, 0], [0, 1e-2]])

nx = noise.WhiteGaussianNoise(state_noise_cov)
ny = noise.WhiteGaussianNoise(meas_noise_cov)

# Model
A = 2*numpy.eye(2)
B = numpy.array([[2, 0.1], [-0.2, 3]])
C = numpy.array([[1, 0], [1, -1]])
D = numpy.array([[0, 0.2], [0, 0]])

m = model.LinearModel(A, B, C, D, nx, ny)

# Controller
P = 5
M = 2
Q = numpy.eye(2)
R = numpy.eye(2)
d = numpy.array([1, 1])
e = 0
k = 1.86

K = controller.SMPC(P, M, Q, R, d, e, m, k)

# Step controller
mu0 = numpy.array([1.2, 3])
u0 = numpy.array([10, 0.1])
sigma0 = numpy.array([[2e-1, 1e-3], [4.5e-3, 7e-2]])
# K.step(mu0, u0, sigma0)
