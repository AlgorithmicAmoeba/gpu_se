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
ts = numpy.linspace(0, end_time, end_time*11)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

# CSTR model
X0 = numpy.array([0.55, 450.])
cstr = model.CSTRModel(X0)

# Linear CSTR model
x_ss_guess = [0.48, 412.0]


def f(x_ss):
    temp = cstr.X
    cstr.X = x_ss
    ans = cstr.DEs(numpy.array([0.]))
    cstr.X = temp
    return ans


X_op = numpy.asarray(scipy.optimize.fsolve(f, numpy.array(x_ss_guess)))
U_op = numpy.array([0.])
cstr1 = model.CSTRModel(X_op)
Y_op = cstr1.outputs(U_op)

lin_model = model.LinearModel.create_LinearModel(cstr, X_op, U_op, dt_control)
# Noise
lin_model.state_noise = noise.WhiteGaussianNoise(covariance=numpy.array([[1e-6, 0], [0, 0.001]]))
lin_model.measurement_noise = noise.WhiteGaussianNoise(covariance=numpy.array([[1e-3, 0], [0, 10]]))

# set point
r = numpy.array([0])

# Controller parameters
P = 150
M = 150
Q = numpy.diag([1e4])
R = numpy.diag([1e-5])

# Bounds
u_bounds = [numpy.array([-1000, 1000]) - U_op[0]]
#
u_step_bounds = [numpy.array([-100, 100])]

K = controller.SMPC(P, M, Q, R, lin_model, r)

# Controller initial params
us = [numpy.zeros_like(U_op)]
xs = [cstr.X.copy()]
ys = [cstr.outputs(us[-1])]

t_next = 0
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        # du = K.step(xs[-1] - X_op, us[-1] - U_op, ys[-1] - Y_op)
        # u = us[-1] + du
        u = controller.mpc_lqr(xs[-1]-X_op, P, lin_model.A, lin_model.B, numpy.diag([1e4, 0]),
                               R, numpy.array([0, 0]), U_op)
        us.append(u)
        t_next += dt_control
    else:
        us.append(us[-1])
    cstr.step(dt, us[-1])
    ys.append(cstr.outputs((us[-1])))
    xs.append(cstr.X.copy())

ys = numpy.array(ys)
us = numpy.array(us)
xs = numpy.array(xs)

plt.subplot(2, 2, 1)
plt.plot(ts, ys)

plt.subplot(2, 2, 2)
plt.plot(ts, xs[:, 1])

plt.subplot(2, 2, 3)
plt.plot(ts, us[:, 0]/60)
plt.show()
