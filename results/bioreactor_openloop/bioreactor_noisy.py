import numpy
import tqdm
import matplotlib.pyplot as plt
import sim_base

# Simulation set-up
end_time = 300
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]

bioreactor, lin_model, _, _ = sim_base.get_parts()
state_pdf, measurement_pdf = sim_base.get_noise()


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
    outputs[lin_model.outputs] += measurement_pdf.draw().get()
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
plt.plot(ts, us[:, lin_model.inputs[1]])
plt.title(r'$F_{m, in}$')

plt.subplot(2, 3, 5)
plt.plot(ts, us[:, lin_model.inputs[0]])
plt.title(r'$F_{G, in}$')

plt.suptitle('Openloop bioreactor with noise')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('noisy.pdf')
plt.show()
