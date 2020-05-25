import numpy
import tqdm
import matplotlib.pyplot as plt
import sim_base

# Simulation set-up
end_time = 300
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]

bioreactor, _, _, _ = sim_base.get_parts()

select_inputs = [0, 2]
select_outputs = [0, 2]

# Initial values
us = [numpy.array([0.04, 5/180, 0.1])]
xs = [bioreactor.X.copy()]
ys = [bioreactor.outputs(us[-1])]

for t in tqdm.tqdm(ts[1:]):
    us.append(us[-1])

    bioreactor.step(dt, us[-1])
    outputs = bioreactor.outputs(us[-1])
    ys.append(outputs.copy())
    xs.append(bioreactor.X.copy())

ys = numpy.array(ys)
us = numpy.array(us)
xs = numpy.array(xs)

plt.subplot(2, 3, 1)
plt.plot(ts, ys[:, 2])
plt.legend(['true'])
plt.title(r'$C_{FA}$')

plt.subplot(2, 3, 2)
plt.plot(ts, ys[:, 0])
plt.legend(['true'])
plt.title(r'$C_{G}$')

plt.subplot(2, 3, 3)
plt.plot(ts, ys[:, 3])
plt.title(r'$C_{E}$')

plt.subplot(2, 3, 4)
plt.plot(ts, us[:, select_inputs[1]])
plt.title(r'$F_{m, in}$')

plt.subplot(2, 3, 5)
plt.plot(ts, us[:, select_inputs[0]])
plt.title(r'$F_{G, in}$')

plt.suptitle('Openloop bioreactor without noise')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('no_noise.pdf')
plt.show()
