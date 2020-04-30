import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel
import noise
import scipy.optimize
import scipy.signal

# Simulation set-up
end_time = 80
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

# CSTR model
X0 = numpy.array([0.6, 420.])
cstr = model.CSTRModel(X0)

# Linear CSTR model
x_ss_guess = [0.21, 467.0]


def f(x_ss):
    temp = cstr.X
    cstr.X = x_ss
    ans = cstr.DEs(numpy.array([0.0]))
    cstr.X = temp
    return ans


X_op = numpy.asarray(scipy.optimize.fsolve(f, numpy.array(x_ss_guess)))
U_op = numpy.array([0.])

lin_model = model.LinearModel.create_LinearModel(cstr, X_op, U_op, dt_control)
# Noise
lin_model.state_noise = noise.WhiteGaussianNoise(covariance=numpy.array([[1e-6, 0], [0, 0.001]]))
lin_model.measurement_noise = noise.WhiteGaussianNoise(covariance=numpy.array([[1e-3, 0], [0, 10]]))

# set point
r = X_op - X_op

# Controller parameters
P = 150
M = 150
Q = numpy.array([[10000, 0], [0, 0]])
R = numpy.eye(1) * 1e-5
D = numpy.array([[10, 1]])
e = numpy.array([412])

# Bounds
x_bounds = [numpy.array([0, 5]) - X_op[0], numpy.array([0, 600]) - X_op[1]]

K = controller.SMPC(P, M, Q, R, D, e, lin_model, r, x_bounds)

# Controller initial params
sigma0 = numpy.zeros((2, 2))

ys = [numpy.array([0.55, 450])]
us = [numpy.zeros_like(U_op)]
xs = [numpy.array([0.55, 450])]

t_next = dt_control
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        us.append(K.step(xs[-1] - X_op, sigma0)+U_op)
        t_next += dt_control
    else:
        us.append(us[-1])
    ys.append(cstr.step(dt, us[-1]))
    xs.append(ys[-1])

ys = numpy.array(ys)
us = numpy.array(us)

plt.plot(ts, ys[:, 0])
plt.show()

plt.plot(ts, ys[:, 1])
plt.show()

plt.plot(ts, us[:, 0])
plt.show()
