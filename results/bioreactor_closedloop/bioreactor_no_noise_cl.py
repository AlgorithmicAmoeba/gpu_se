import numpy
import tqdm
import matplotlib.pyplot as plt
import sim_base

# Simulation set-up
end_time = 50
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

bioreactor, lin_model, K, _ = sim_base.get_parts(dt_control=dt_control)

# Initial values
us = [numpy.array([0.06, 5/180, 0.2])]
xs = [bioreactor.X.copy()]
ys = [bioreactor.outputs(us[-1])]

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
    outputs = bioreactor.outputs(us[-1])
    ys.append(outputs.copy())
    xs.append(bioreactor.X.copy())

ys = numpy.array(ys)
us = numpy.array(us)
xs = numpy.array(xs)
biass = numpy.array(biass)

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
plt.savefig('no_noise_cl.pdf')
plt.show()
