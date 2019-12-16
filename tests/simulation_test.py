import numpy
import model
import controller
import noise

# Simulation set-up
ts = numpy.linspace(0, 10, 100)
dt = ts[1]

# CSTR model
X0 = numpy.array([1., 320.])
cstr = model.CSTRModel(X0)

# Noise
state_noise_cov = numpy.array([[1e-1, 0], [0, 7e-2]])
meas_noise_cov = numpy.array([[6e-2, 0], [0, 1e-2]])

nx = noise.WhiteGaussianNoise(state_noise_cov)
ny = noise.WhiteGaussianNoise(meas_noise_cov)

# Linear CSTR model
X_op = X0
input_op = numpy.array([0., 0.1])

lin_model = model.create_LinearModel(cstr, X_op, input_op, dt)
lin_model.state_noise = nx
lin_model.measurement_noise = ny

# Controller parameters
P = 20
M = 5
Q = numpy.eye(2)
R = numpy.eye(2)
d = numpy.array([0, 0])
e = 0
k = 1.86

K = controller.SMPC(P, M, Q, R, d, e, lin_model, k)

# Controller initial params
mu0 = X0
u0 = input_op
sigma0 = numpy.zeros((2, 2))
r = numpy.array([0.3, 400])

ys = [X0]
us = [numpy.zeros_like(u0)]

for t in ts[1:]:
    us.append(K.step(ys[-1], us[-1], sigma0, r))
    ys.append(cstr.step(dt, us[-1]))


