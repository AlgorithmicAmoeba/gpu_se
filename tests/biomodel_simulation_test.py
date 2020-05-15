import numpy
import scipy.optimize
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel


def find_SS(U_op, X0):
    bioreactor_SS = model.Bioreactor(X0=[], high_N=False)

    def fun(x_ss):
        temp = bioreactor_SS.X
        bioreactor_SS.X = x_ss
        ans = bioreactor_SS.DEs(U_op)
        bioreactor_SS.X = temp
        return ans

    return scipy.optimize.fsolve(fun, X0)


# Simulation set-up
end_time = 50
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

# Bioreactor

bioreactor = model.Bioreactor(
    X0=find_SS(
        numpy.array([0.06, 5/180, 0.2]),
        #            Ng,         Nx,      Nfa, Ne, Nh
        numpy.array([0.26/180, 0.64/24.6, 1/116, 0, 0])
    ),
    high_N=False
)

# Linear model
lin_model = model.LinearModel.create_LinearModel(
    bioreactor,
    x_bar=find_SS(
        numpy.array([0.04, 5/180, 0.1]),
        #           Ng,         Nx,      Nfa, Ne, Nh
        numpy.array([0.26/180, 0.64/24.6, 1/116, 0, 0])
    ),
    #          Fg_in (L/h), Cg (mol/L), Fm_in (L/h)
    u_bar=numpy.array([0.04, 5/180, 0.1]),
    T=dt_control
)
#  Select states, outputs and inputs for MPC
#        Cfa
states = [0, 2]
#     Fm_in
inputs = [0, 2]
#         Cfa
outputs = [0, 2]

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

# set point
r = lin_model.yn2d(numpy.array([0.28, 0.85]))

# Controller parameters
P = 200
M = 160
Q = numpy.diag([1e2, 1e3])
R = numpy.diag([1e0, 1e0])

u_bounds = [numpy.array([0, numpy.inf]) - lin_model.u_bar[0],
            numpy.array([0, numpy.inf]) - lin_model.u_bar[1]]
K = controller.LQR(P, M, Q, R, lin_model, r, u_bounds=u_bounds)

# Controller initial params
# Non-linear
us = [numpy.array([0.06, 5/180, 0.2])]
ys = [bioreactor.outputs(us[-1])]
xs = [bioreactor.X.copy()]

biass = []

t_next = 0
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        U_temp = us[-1].copy()
        if K.y_predicted is not None:
            biass.append(lin_model.yn2d(ys[-1][outputs]) - K.y_predicted)

        u = K.step(lin_model.xn2d(xs[-1][states]), lin_model.un2d(us[-1][inputs]), lin_model.yn2d(ys[-1][outputs]))
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
biass = numpy.array(biass)

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
plt.plot(
    numpy.arange(dt_control, end_time, dt_control),
    biass
)
plt.title('bias')

plt.subplot(2, 3, 5)
plt.plot(ts, ys[:, 3])
plt.title('Ce')

plt.subplot(2, 3, 6)
plt.plot(ts, us[:, inputs[1]])
plt.title('Fg_in')
plt.show()
