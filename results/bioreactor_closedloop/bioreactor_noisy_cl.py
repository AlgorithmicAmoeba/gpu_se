import numpy
import tqdm
import matplotlib.pyplot as plt
import controller
import model.LinearModel
import gpu_funcs.MultivariateGaussianSum

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

state_pdf = gpu_funcs.MultivariateGaussianSum(
    means=numpy.zeros(shape=(1, 5)),
    covariances=numpy.diag([1e-10, 1e-8, 1e-9, 1e-9, 1e-9])[numpy.newaxis, :, :],
    weights=numpy.array([1.])
)
measurement_pdf = gpu_funcs.MultivariateGaussianSum(
    means=numpy.array([[1e-4, 0],
                       [0, -1e-4]]),
    covariances=numpy.array([[[6e-5, 0],
                              [0, 8e-5]],

                             [[5e-5, 1e-5],
                              [1e-5, 7e-5]]]),
    weights=numpy.array([0.85, 0.15])
)

# Initial values
us = [numpy.array([0.06, 5/180, 0.2])]
xs = [bioreactor.X.copy()]
ys = [bioreactor.outputs(us[-1])]
ys_meas = [bioreactor.outputs(us[-1])]

biass = []

t_next = 0
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        U_temp = us[-1].copy()
        if K.y_predicted is not None:
            biass.append(lin_model.yn2d(ys_meas[-1]) - K.y_predicted)

        u = K.step(lin_model.xn2d(xs[-1]), lin_model.un2d(us[-1]), lin_model.yn2d(ys_meas[-1]))
        U_temp[lin_model.inputs] = lin_model.ud2n(u)
        us.append(U_temp.copy())
        t_next += dt_control
    else:
        us.append(us[-1])

    bioreactor.step(dt, us[-1])
    bioreactor.X += state_pdf.draw().get()
    outputs = bioreactor.outputs(us[-1])
    ys.append(outputs.copy())
    outputs[lin_model.outputs] += measurement_pdf.draw().get()
    ys_meas.append(outputs)
    xs.append(bioreactor.X.copy())

ys = numpy.array(ys)
ys_meas = numpy.array(ys_meas)
us = numpy.array(us)
xs = numpy.array(xs)
biass = numpy.array(biass)

plt.subplot(2, 3, 1)
plt.plot(ts, ys_meas[:, 2])
plt.plot(ts, ys[:, 2])
plt.legend(['measured', 'true'])
plt.title(r'$C_{FA}$')

plt.subplot(2, 3, 2)
plt.plot(ts, ys_meas[:, 0])
plt.plot(ts, ys[:, 0])
plt.legend(['measured', 'true'])
plt.title(r'$C_{G}$')

plt.subplot(2, 3, 3)
plt.plot(ts, ys_meas[:, 3])
plt.title(r'$C_{E}$')

plt.subplot(2, 3, 4)
plt.plot(ts, us[:, lin_model.inputs[0]])
plt.title(r'$F_{m, in}$')

plt.subplot(2, 3, 5)
plt.plot(ts, us[:, lin_model.inputs[1]])
plt.title(r'$F_{G, in}$')

plt.subplot(2, 3, 6)
plt.plot(
    numpy.arange(dt_control, end_time, dt_control),
    biass
)
plt.legend([r'$C_G$', r'$C_{FA}$'])
plt.title('bias')

plt.suptitle('Closedloop bioreactor with noise')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('noisy_cl.pdf')
plt.show()
