import numpy
import tqdm
import matplotlib.pyplot as plt
import model.LinearModel
import sim_base

# Simulation set-up
end_time = 50
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]
dt_control = 1
assert dt <= dt_control

bioreactor, lin_model, K, pf = sim_base.get_parts(
    dt_control=dt_control,
    N_particles=2**15
)
state_pdf, measurement_pdf = sim_base.get_noise()

# Initial values
us = [numpy.array([0.06, 0.2])]
xs = [bioreactor.X.copy()]
ys = [bioreactor.outputs(us[-1])]
ys_meas = [bioreactor.outputs(us[-1])]
ys_pf = [
    model.Bioreactor.static_outputs(
            pf.point_estimate(),
            us[-1]
        )
]

biass = []

t_next = 0
for t in tqdm.tqdm(ts[1:]):
    if t > t_next:
        U_temp = us[-1].copy()
        if K.y_predicted is not None:
            biass.append(lin_model.yn2d(ys_meas[-1]) - K.y_predicted)

        pf.update(us[-1], ys_meas[-1][lin_model.outputs])
        pf.resample()
        x_pf = (pf.weights @ pf.particles).get()
        try:
            u = K.step(
                lin_model.xn2d(x_pf),
                lin_model.un2d(us[-1]),
                lin_model.yn2d(ys_meas[-1])
            )
        except:
            u = numpy.array([0.06, 0.2])
        U_temp[lin_model.inputs] = lin_model.ud2n(u)
        us.append(U_temp.copy())
        t_next += dt_control
    else:
        us.append(us[-1])

    bioreactor.step(dt, us[-1])
    bioreactor.X += state_pdf.draw().get().squeeze()
    outputs = bioreactor.outputs(us[-1])
    ys.append(outputs.copy())
    outputs[lin_model.outputs] += measurement_pdf.draw().get().squeeze()
    ys_meas.append(outputs)
    xs.append(bioreactor.X.copy())

    pf.predict(us[-1], dt)
    ys_pf.append(
        numpy.array(
            model.Bioreactor.static_outputs(
                (pf.weights @ pf.particles).get(),
                us[-1]
            )
        )
    )

ys = numpy.array(ys)
ys_meas = numpy.array(ys_meas)
us = numpy.array(us)
xs = numpy.array(xs)
ys_pf = numpy.array(ys_pf)
biass = numpy.array(biass)

plt.subplot(2, 3, 1)
plt.plot(ts, ys_meas[:, 2])
plt.plot(ts, ys_pf[:, 1])
plt.plot(ts, ys[:, 2])
plt.axhline(lin_model.yd2n(K.ysp)[1], color='red')
plt.legend(['measured', 'predicted', 'true'])
plt.title('Cfa')

plt.subplot(2, 3, 2)
plt.plot(ts, ys_meas[:, 0])
plt.plot(ts, ys_pf[:, 0])
plt.plot(ts, ys[:, 0])
plt.axhline(lin_model.yd2n(K.ysp)[0], color='red')
plt.legend(['measured', 'predicted', 'true'])
plt.title('Cg')

plt.subplot(2, 3, 3)
plt.plot(ts, ys_meas[:, 3])
plt.title('Ce')

plt.subplot(2, 3, 4)
plt.plot(ts, us[:, lin_model.inputs[1]])
plt.title('Fm_in')

plt.subplot(2, 3, 5)
plt.plot(ts, us[:, lin_model.inputs[0]])
plt.title('Fg_in')

plt.subplot(2, 3, 6)
plt.plot(
    numpy.arange(dt_control, end_time, dt_control),
    biass
)
plt.title('bias')

plt.show()
