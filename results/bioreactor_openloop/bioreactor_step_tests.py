import numpy
import tqdm
import matplotlib.pyplot as plt
import sim_base
import joblib
import itertools
import warnings
import matplotlib
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

memory = joblib.Memory('cache/')


@memory.cache
def step_test(percent, dt):
    # Simulation set-up
    end_time = 300
    ts = numpy.linspace(0, end_time, int(end_time//dt))

    bioreactor, lin_model, _, _ = sim_base.get_parts()

    # Initial values
    u = numpy.array([0.06, 0.2])
    u *= percent
    ys = [bioreactor.outputs(u)]

    for _ in tqdm.tqdm(ts[1:]):

        bioreactor.step(dt, u)
        outputs = bioreactor.outputs(u)
        ys.append(outputs.copy())

    ys = numpy.array(ys)
    return ts, ys


def plot():
    percents = numpy.array([0.7, 0.8, 1, 1.2, 1.3])
    dts = [0.1]

    u = numpy.array([0.06, 0.2])
    plt.figure(figsize=(6.4*2, 4.8))
    for p1, p2, dt in itertools.product(percents, percents, dts):
        percent = numpy.array([p1, p2])
        ts, ys = step_test(percent, dt)
        # u_i = u * percent

        plt.subplot(1, 2, 1)
        plt.plot(ts, ys[:, 2])
        plt.title(r'$C_{FA}$')
        plt.ylabel(r'$\frac{mol}{L}$')
        plt.xlabel(r't ($min$)')

        plt.subplot(1, 2, 2)
        plt.plot(ts, ys[:, 0])
        plt.title(r'$C_{G}$')
        plt.ylabel(r'$\frac{mol}{L}$')
        plt.xlabel(r't ($min$)')

        # plt.subplot(2, 2, 3)
        # plt.axhline(u_i[1])
        # plt.title(r'$F_{M, in}$')
        # plt.ylabel(r'$\frac{L}{min}$')
        # plt.xlabel(r't ($min$)')
        #
        # plt.subplot(2, 2, 4)
        # plt.axhline(u_i[0])
        # plt.title(r'$F_{G, in}$')
        # plt.ylabel(r'$\frac{L}{min}$')
        # plt.xlabel(r't ($min$)')

        plt.suptitle('Step tests')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('batch_step_test.pdf')
    plt.show()


plot()
