import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel

# Simulation set-up
end_time = 50
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

# Bioreactor
bioreactor = model.Bioreactor(
    X0=model.Bioreactor.find_SS(
        numpy.array([0.06, 5/180, 0.2]),
        #            Ng,         Nx,      Nfa, Ne, Nh
        numpy.array([0.26/180, 0.64/24.6, 1/116, 0, 0])
    ),
    high_N=False
)

# Linear model
lin_model = model.LinearModel.create_LinearModel(
    bioreactor,
    x_bar=model.Bioreactor.find_SS(
        numpy.array([0.04, 5/180, 0.1]),
        #           Ng,         Nx,      Nfa, Ne, Nh
        numpy.array([0.26/180, 0.64/24.6, 1/116, 0, 0])
    ),
    #          Fg_in (L/h), Cg (mol/L), Fm_in (L/h)
    u_bar=numpy.array([0.04, 5/180, 0.1]),
    T=dt_control
)
#  Select states, outputs and inputs for MPC
lin_model.select_subset(
    states=[0, 2],  # Cg, Cfa
    inputs=[0, 2],  # Fg_in, Fm_in
    outputs=[0, 2],  # Cg, Cfa
)

# Controller
K = controller.MPC(
    P=200,
    M=160,
    Q=numpy.diag([1e2, 1e3]),
    R=numpy.diag([1, 1]),
    lin_model=lin_model,
    ysp=lin_model.yn2d(numpy.array([0.28, 0.85]), subselect=False),
    u_bounds=[
        numpy.array([0, numpy.inf]) - lin_model.u_bar[0],
        numpy.array([0, numpy.inf]) - lin_model.u_bar[1]
    ]
)

# Controller initial params
us = [numpy.array([0.06, 5/180, 0.2])]
ys = [bioreactor.outputs(us[-1])]
xs = [bioreactor.X.copy()]

biass = []

t_next = 0
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        U_temp = us[-1].copy()
        if K.y_predicted is not None:
            biass.append(lin_model.yn2d(ys[-1]) - K.y_predicted)

        u = K.step(lin_model.xn2d(xs[-1]), lin_model.un2d(us[-1]), lin_model.yn2d(ys[-1]))
        U_temp[lin_model.inputs] = lin_model.ud2n(u)
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
plt.plot(ts, us[:, lin_model.inputs[0]])
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
plt.plot(ts, us[:, lin_model.inputs[1]])
plt.title('Fg_in')
plt.show()
