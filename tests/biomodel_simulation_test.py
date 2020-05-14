import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel
import noise

# Simulation set-up
end_time = 1000
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

# Bioreactor model
#                    Ng,        Nx,         Nfa, Ne, Na, Nb, Nh, V, T
X0 = numpy.array([0.186/180, 0.639773/24.6, 0.86/116, 0, 0])
bioreactor = model.Bioreactor(X0)
#                    Ng,        Nx,         Nfa, Ne, Na, Nb, Nh, V, T
X_op = numpy.array([0.17992/180, 0.639773/24.6, 0.764/116, 0, 0])
# Inputs
#          Fg_in (L/h), Cg (mol/L), Fm_in (L/h)
U_op = numpy.array([4.65, 1/180, 0.5])

lin_model = model.LinearModel.create_LinearModel(bioreactor, X_op, U_op, dt_control)
#  Select states, outputs and inputs for MPC
#        Nfa
states = [2]
#     Fm_in
inputs = [2]
#         Cfa
outputs = [2]

lin_model.A = lin_model.A[states][:, states]
lin_model.B = lin_model.B[states][:, inputs]
lin_model.C = lin_model.C[outputs][:, states]
lin_model.D = lin_model.D[outputs][:, inputs]
lin_model.x_bar = lin_model.x_bar[states]
lin_model.u_bar = lin_model.u_bar[inputs]
lin_model.f_bar = lin_model.f_bar[states]
lin_model.y_bar = lin_model.y_bar[outputs]
lin_model.Nx = len(states)
lin_model.Ni = len(inputs)
lin_model.No = len(outputs)
bioreactor.high_N = False

bioreactor1 = model.Bioreactor(X_op)
Y_op = bioreactor1.outputs(U_op)[outputs]

# Noise
lin_model.state_noise = noise.WhiteGaussianNoise(covariance=numpy.diag([1e-4, 1e-8, 1e-8, 1e-2]))
lin_model.measurement_noise = noise.WhiteGaussianNoise(covariance=numpy.diag([1e-3, 1e-2]))

# set point
r = numpy.array([3]) - Y_op

# Controller parameters
P = 100
M = 80
Q = numpy.diag([1e3])
R = numpy.diag([1e0])

LQR = controller.LQR(P, M, Q, R, lin_model, r)

# Controller initial params
# Non-linear
us = [numpy.array([4.65, 1/180, 0.])]
ys = [bioreactor.outputs(us[-1])]
xs = [bioreactor.X.copy()]

t_next = 0
count = 0
not_done = True
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        U_temp = us[-1].copy()
        u = LQR.step(lin_model.xn2d(xs[-1][states]), lin_model.un2d(us[-1][inputs]), lin_model.yn2d(ys[-1][outputs]))
        U_temp[inputs] = lin_model.ud2n(u)
        us.append(U_temp.copy())
        t_next += dt_control
    else:
        us.append(us[-1])

    bioreactor.step(dt, us[-1])
    ys.append(bioreactor.outputs(us[-1]))
    xs.append(bioreactor.X.copy())

ys = numpy.array(ys)
us = numpy.array(us)
xs = numpy.array(xs)

plt.subplot(2, 3, 1)
plt.plot(ts, ys[:, 2])
plt.title('Cfa')

plt.subplot(2, 3, 2)
plt.plot(ts, us[:, inputs[0]])
plt.title('Fm_in')

plt.subplot(2, 3, 3)
plt.plot(ts, ys[:, 0])
plt.title('Cg')

plt.subplot(2, 3, 4)
plt.plot(ts, ys[:, 1])
plt.title('Cx')

plt.subplot(2, 3, 5)
plt.plot(ts, ys[:, 3])
plt.title('Ce')

plt.subplot(2, 3, 6)
plt.plot(ts, ys[:, 4])
plt.title('Ch')
plt.show()
