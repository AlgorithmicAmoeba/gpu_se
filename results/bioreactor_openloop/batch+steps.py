import numpy
import tqdm
import matplotlib.pyplot as plt
import sim_base
import model

# Simulation set-up
end_time = 800
ts = numpy.linspace(0, end_time, end_time*10)
dt = ts[1]

bioreactor = model.Bioreactor(
    #                Ng,         Nx,      Nfa, Ne, Nh
    X0=numpy.array([3 / 180, 1e-3 / 24.6, 0 / 116, 0., 0.]),
    high_N=True
)

select_inputs = [0, 2]  # Fg_in, Fm_in
select_outputs = [0, 2]  # Cg, Cfa

state_pdf, measurement_pdf = sim_base.get_noise()


# Initial values
us = [numpy.array([0., 5/180, 0.])]
xs = [bioreactor.X.copy()]
ys = [bioreactor.outputs(us[-1])]
ys_meas = [bioreactor.outputs(us[-1])]

not_cleared = True
Cx = numpy.NaN
for t in tqdm.tqdm(ts[1:]):
    if t < 25:
        us.append(numpy.array([0., 5/180, 0.]))
    elif t < 200:
        if not_cleared:
            bioreactor.X[[0, 2, 3, 4]] = 0
            not_cleared = False
            bioreactor.high_N = False
            Cx = bioreactor.X[1]

        # (L/min) = (gG/gX/min) (molG/gG) (molX/Lv) (gX/molX) (Lv) (L/molG)
        glucose = 0.3 / 180 * Cx * 24.6 * 1 / (5/180)
        us.append(numpy.array([glucose, 5/180, 0.]))
    elif t < 500:
        # (L/min) = (gG/gX/min) (molG/gG) (molX/Lv) (gX/molX) (Lv) (L/molG)
        glucose = 0.45 / 180 * Cx * 24.6 * 1 / (5/180)
        us.append(numpy.array([glucose, 5/180, 0.]))
    elif t < 700:
        # (L/min) = (gG/gX/min) (molG/gG) (molX/Lv) (gX/molX) (Lv) (L/molG)
        glucose = 0.6 / 180 * Cx * 24.6 * 1 / (5/180)
        us.append(numpy.array([glucose, 5/180, 0.]))
    else:
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


def add_time_lines():
    for time in [25, 200, 500, 700]:
        plt.axvline(time, color='black', alpha=0.4)
    plt.xlim([0, ts[-1]])


plt.subplot(2, 3, 1)
plt.plot(ts, ys_meas[:, 2])
plt.title(r'$C_{FA}$')
add_time_lines()

plt.subplot(2, 3, 2)
plt.plot(ts, ys_meas[:, 0])
plt.title(r'$C_{G}$')
add_time_lines()

plt.subplot(2, 3, 3)
plt.plot(ts, ys_meas[:, 3])
plt.title(r'$C_{E}$')
add_time_lines()

plt.subplot(2, 3, 4)
plt.plot(ts, us[:, select_inputs[1]])
plt.title(r'$F_{m, in}$')
add_time_lines()

plt.subplot(2, 3, 5)
plt.plot(ts, us[:, select_inputs[0]])
plt.title(r'$F_{G, in}$')
plt.xlim([0, ts[-1]])
for c in [0.4, 0.5]:
    glucose = c / 180 * bioreactor.X[1] * 24.6 * 1 / (5/180)
    plt.axhline(glucose, color='green', alpha=0.4)

plt.subplot(2, 3, 6)
plt.plot(ts, ys[:, 1])
plt.title(r'$C_{X}$')
add_time_lines()

plt.suptitle('Openloop growth and production run')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('batch+steps_noisy.pdf')
plt.show()
