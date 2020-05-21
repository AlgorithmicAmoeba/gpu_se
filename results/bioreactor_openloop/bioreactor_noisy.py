import numpy
import tqdm
import matplotlib.pyplot as plt
import model.LinearModel
import gpu_funcs.MultivariateGaussianSum

# Simulation set-up
end_time = 300
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]

# Bioreactor
bioreactor = model.Bioreactor(
    X0=model.Bioreactor.find_SS(
        numpy.array([0.06, 5/180, 0.2]),
        #            Ng,         Nx,      Nfa, Ne, Nh
        numpy.array([0.26/180, 0.64/24.6, 1/116, 0, 0])
    ),
    high_N=False
)

select_inputs = [0, 2]
select_outputs = [0, 2]

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
us = [numpy.array([0.04, 5/180, 0.1])]
xs = [bioreactor.X.copy()]
ys = [bioreactor.outputs(us[-1])]
ys_meas = [bioreactor.outputs(us[-1])]

for t in tqdm.tqdm(ts[1:]):
    us.append(us[-1])

    bioreactor.step(dt, us[-1])
    bioreactor.X += state_pdf.draw().get()
    outputs = bioreactor.outputs(us[-1])
    ys.append(outputs.copy())
    outputs[select_outputs] += measurement_pdf.draw().get()
    ys_meas.append(outputs)
    xs.append(bioreactor.X.copy())

ys = numpy.array(ys)
ys_meas = numpy.array(ys_meas)
us = numpy.array(us)
xs = numpy.array(xs)

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
plt.plot(ts, us[:, select_inputs[0]])
plt.title(r'$F_{m, in}$')

plt.subplot(2, 3, 5)
plt.plot(ts, us[:, select_inputs[1]])
plt.title(r'$F_{G, in}$')

plt.suptitle('Openloop bioreactor with noise')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('noisy.pdf')
plt.show()
