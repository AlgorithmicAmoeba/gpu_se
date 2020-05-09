import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel
import noise

# Simulation set-up
end_time = 100
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]
dt_control = 1*dt
assert dt <= dt_control

# Bioreactor model
#                    Ng,        Nx,         Nfa, Ne, Na, Nb, Nh, V, T
X0 = numpy.array([0.28/180, 0.639773/24.6, 1/116, 0, 1e-5, 0, 4.857e-3, 1.077, 35])
bioreactor = model.Bioreactor(X0, pH_calculations=True)
#                    Ng,        Nx,         Nfa, Ne, Na, Nb, Nh, V, T
X_op = numpy.array([0.28/180, 0.639773/24.6, 2/116, 0, 0, 3.43e-2, 4.857e-3, 1.077, 35])
# Inputs
Cn_in = 0.625 * 10 / 60  # (g/L) / (g/mol) = mol/L
CgFg = 0.23

Cg_in = 314.19206 / 180  # (g/L) / (g/mol) = mol/L
Ca_in = 0.1  # mol/L
Cb_in = 0.1  # mol/L
Fm_in = 0.1

Fg_in = CgFg / 180 / Cg_in  # (g/h) / (g/mol) / (mol/L) = L/h
Fn_in = 0.625 / 1000 / Cn_in / 60  # (mg/h) / (mg/g) / (mol/L) / (g/mol) = L/h
Fa_in = 6e-9
Fb_in = 6e-9  # L/h
F_out = Fg_in + Fn_in + Fa_in + Fb_in + Fm_in

T_amb = 25
Q = 5 / 9

U_op = numpy.array([Fg_in, Cg_in, Fa_in, Ca_in, Fb_in, Cb_in, Fm_in, F_out, T_amb, Q])

lin_model = model.LinearModel.create_LinearModel(bioreactor, X_op, U_op, dt_control)
#  Select states, outputs and inputs for MPC
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

bioreactor1 = model.Bioreactor(X_op, pH_calculations=True)
Y_op = bioreactor1.outputs(U_op)[outputs]

# Noise
lin_model.state_noise = noise.WhiteGaussianNoise(covariance=numpy.diag([1e-4, 1e-8, 1e-8, 1e-2]))
lin_model.measurement_noise = noise.WhiteGaussianNoise(covariance=numpy.diag([1e-3, 1e-2]))

# set point
r = numpy.array([3, 5]) - Y_op

# Controller parameters
P = 3
M = 2
Q = numpy.diag([1e0, 1e0])
R = numpy.diag([1e0, 1e0, 1e0])

# Bounds
# x_bounds = [numpy.array([0, 5]) - X_op[0], numpy.array([0, 600]) - X_op[1]]
u_bounds = [numpy.array([-0.75, 1]) - U_op[[inputs[0]]],
            numpy.array([-0.75, 1]) - U_op[[inputs[1]]],
            numpy.array([-0.75, 1]) - U_op[[inputs[2]]]]
#
u_step_bounds = [numpy.array([-0.1, 0.1]),
                 numpy.array([-0.1, 0.1]),
                 numpy.array([-0.1, 0.1])]

K = controller.SMPC(P, M, Q, R, lin_model, r, u_bounds=u_bounds, u_step_bounds=u_step_bounds)

# Controller initial params
# Non-linear
# us = [U_op]
# ys = [bioreactor.outputs(us[-1])[outputs]]
# xs = [bioreactor.X]
# K.x_predicted = xs[-1][states] - lin_model.x_bar

# Linear
us = [U_op]
xs = [bioreactor.X[states]]
ys = [lin_model.C @ (xs[-1] - lin_model.x_bar) + lin_model.D @ (us[-1][inputs] - lin_model.u_bar)
      + lin_model.g_bar]

t_next = 0
count = 0
not_done = True
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        U_temp = us[-1].copy()
        # For nonlinear model
        # du = K.step(xs[-1][states] - lin_model.x_bar, us[-1][inputs] - U_op[inputs], ys[-1] - Y_op)
        # For linear model
        du = K.step(xs[-1] - lin_model.x_bar, us[-1][inputs] - lin_model.u_bar, ys[-1] - Y_op)

        U_temp[inputs] = U_temp[inputs] + du
        U_temp[-3] = Fg_in + Fn_in + sum(U_temp[inputs])
        us.append(U_temp.copy())
        t_next += dt_control
    else:
        us.append(us[-1])

    # Linear model
    xs.append(lin_model.A @ (xs[-1] - lin_model.x_bar) + lin_model.B @ (us[-1][inputs] - lin_model.u_bar)
              + lin_model.x_bar)

    if t > 50 and count < 1e5:  # Disturbance for linear model
        xs[-1][0] += 0.001
        count += 1

    if t > 25 and not_done:
        lin_model.A *= 0.9
        not_done = False

    ys.append(lin_model.C @ (xs[-1] - lin_model.x_bar) + lin_model.D @ (us[-1][inputs] - lin_model.u_bar)
              + lin_model.g_bar)
    # Non-linear model
    # bioreactor.step(dt, us[-1])
    # ys.append(bioreactor.outputs(us[-1])[outputs])
    # xs.append(bioreactor.X)

    if t > 90:
        pass

ys = numpy.array(ys)
us = numpy.array(us)

plt.subplot(2, 3, 1)
plt.plot(ts, ys[:, 0])

plt.subplot(2, 3, 2)
plt.plot(ts, ys[:, 1])

plt.subplot(2, 3, 4)
plt.plot(ts, us[:, inputs[0]])

plt.subplot(2, 3, 5)
plt.plot(ts, us[:, inputs[1]])

plt.subplot(2, 3, 6)
plt.plot(ts, us[:, inputs[2]])
plt.show()
