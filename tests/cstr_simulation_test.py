import numpy
import tqdm
import matplotlib.pyplot as plt
import model
import controller
import noise

# Simulation set-up
end_time = 1000
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]

# CSTR model
X0 = numpy.array([0.6, 420.])
cstr = model.CSTRModel(X0)

# Noise
state_noise_cov = numpy.array([[1e-6, 0], [0, 0.001]])
meas_noise_cov = numpy.array([[1e-3, 0], [0, 10]])

nx = noise.WhiteGaussianNoise(state_noise_cov)
ny = noise.WhiteGaussianNoise(meas_noise_cov)

# set point
r = numpy.array([0.4893, 412])

# Linear CSTR model
X_op = r
input_op = numpy.array([0., 0.1])

lin_model = model.create_LinearModel(cstr, X_op, input_op, dt)
lin_model.state_noise = nx
lin_model.measurement_noise = ny

# Controller parameters
P = int(10/dt)
M = int(5/dt)
Q = numpy.array([[10000, 0], [0, 1]])
R = numpy.eye(2)
D = numpy.array([[0, 1], [1, 0]])
e = numpy.array([100, 0])
p = 0

# Bounds
x_bounds = [(0, 5), (0, 600)]
u_bounds = [(-300, 300), (0, 3)]
u_step_bounds = [(-10, 10), (-0.1, 0.1)]

# K = controller.SMPC(P, M, Q, R, D, e, lin_model, r, x_bounds, u_bounds, u_step_bounds, p)
K = controller.SMPC(P, M, Q, R, D, e, lin_model, r)

# Controller initial params
mu0 = X0
u0 = input_op
sigma0 = numpy.zeros((2, 2))

ys = [X0]
us = [numpy.zeros_like(u0)]

for t in tqdm.tqdm(ts[1:]):
    us.append(K.step(ys[-1], sigma0))
    ys.append(cstr.step(dt, us[-1]))

ys = numpy.array(ys)
us = numpy.array(us)

plt.plot(ts, ys[:, 0])
plt.show()

plt.plot(ts, ys[:, 1])
plt.show()

plt.plot(ts, us[:, 0])
plt.show()

plt.plot(ts, us[:, 1])
plt.show()
