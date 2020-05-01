import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel
import noise
import scipy.optimize
import scipy.signal

# Simulation set-up
end_time = 1000
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

# Bioreactor model
#                    Ng,        Nx,         Nfa, Ne, Na, Nb, Nh, V, T
X0 = numpy.array([0.28/180, 0.639773/24.6, 1/116, 0, 1e-5, 0, 4.857e-3, 1.077, 35])
bioreactor = model.Bioreactor(X0, pH_calculations=True)
#                    Ng,        Nx,         Nfa, Ne, Na, Nb, Nh, V, T
X_op = numpy.array([0.28/180, 0.639773/24.6, 2/116, 0, 1e-5, 0, 4.857e-3, 1.077, 35])
# Inputs
Cn_in = 0.625 * 10 / 60  # (g/L) / (g/mol) = mol/L
CgFg = 0.23

Cg_in = 314.19206 / 180  # (g/L) / (g/mol) = mol/L
Ca_in = 10  # mol/L
Cb_in = 10  # mol/L
Fm_in = 0

Fg_in = CgFg / 180 / Cg_in  # (g/h) / (g/mol) / (mol/L) = L/h
Fn_in = 0.625 / 1000 / Cn_in / 60  # (mg/h) / (mg/g) / (mol/L) / (g/mol) = L/h
Fa_in = 6e-9
Fb_in = 6e-9  # L/h
F_out = Fg_in + Fn_in + Fa_in + Fb_in + Fm_in

T_amb = 25
Q = 5 / 9

U_op = numpy.array([Fg_in, Cg_in, Fa_in, Ca_in, Fb_in, Cb_in, Fm_in, F_out, T_amb, Q])

lin_model = model.LinearModel.create_LinearModel(bioreactor, X_op, U_op, dt_control)
#        Nfa, Na, Nb, V
states = [2, 4, 5, 7]
#     Fa_in, Fb_in, Fm_in
inputs = [2, 4, 6]
#      Cfa, pH
outputs = [2, 9]

lin_model.A = lin_model.A[states][:, states]
lin_model.B = lin_model.B[states][:, inputs]
lin_model.C = lin_model.C[outputs][:, states]
lin_model.D = lin_model.D[outputs][:, inputs]
lin_model.x_bar = lin_model.x_bar[states]
lin_model.u_bar = lin_model.u_bar[inputs]
lin_model.f_bar = lin_model.f_bar[states]
lin_model.g_bar = lin_model.g_bar[outputs]
lin_model.Nx = len(states)
lin_model.Ni = len(inputs)
lin_model.No = len(outputs)
bioreactor.high_N = False

# Noise
lin_model.state_noise = noise.WhiteGaussianNoise(covariance=numpy.diag([1e-4, 1e-8, 1e-8, 1e-2]))
lin_model.measurement_noise = noise.WhiteGaussianNoise(covariance=numpy.diag([1e-3, 1e-2]))

# set point
r = (X_op - X_op)[states]

# Controller parameters
P = 150
M = 100
Q = numpy.diag([1e4, 0, 0, 0])
R = numpy.diag([1, 0, 1])
D = numpy.array([[0, 0, 0, 0]])
e = numpy.array([0])

# Bounds
# x_bounds = [numpy.array([0, 5]) - X_op[0], numpy.array([0, 600]) - X_op[1]]
u_bounds = [numpy.array([0, 1e-8]) - U_op[[inputs[0]]],
            numpy.array([0, 1e-8]) - U_op[[inputs[1]]],
            numpy.array([0, 10]) - U_op[[inputs[2]]]]

K = controller.SMPC(P, M, Q, R, D, e, lin_model, r, u_bounds=u_bounds)

# Controller initial params
sigma0 = numpy.zeros((4, 4))

ys = [numpy.array([1/116/1.077, 5])]
us = [numpy.zeros_like(U_op)]
xs = [numpy.array([1/116, 1e-5, 0, 1.077])]

t_next = dt_control
U_temp = U_op.copy()
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        U_temp[inputs] = K.step(xs[-1][states] - X_op[states], sigma0)
        U_temp[-3] = Fg_in + Fn_in + sum(U_temp[inputs])
        us.append(U_temp)
        t_next += dt_control
    else:
        us.append(us[-1])

    # xs.append(lin_model.A @ (xs[-1] - X_op[states]) + lin_model.B @ (us[-1] - U_op)[inputs] + lin_model.x_bar + lin_model.f_bar)
    # ys.append(lin_model.C @ (xs[-1] - X_op[states]) + lin_model.D @ (us[-1] - U_op)[inputs] + lin_model.g_bar)
    bioreactor.step(dt, us[-1])
    ys.append(bioreactor.outputs(us[-1])[outputs])
    xs.append(bioreactor.X)

ys = numpy.array(ys)
us = numpy.array(us)

plt.plot(ts, ys[:, 0]*116)
plt.show()

plt.plot(ts, ys[:, 1])
plt.show()

plt.subplot(1, 3, 1)
plt.plot(ts, us[:, inputs[0]])

plt.subplot(1, 3, 2)
plt.plot(ts, us[:, inputs[1]])

plt.subplot(1, 3, 3)
plt.plot(ts, us[:, inputs[2]])
plt.show()
