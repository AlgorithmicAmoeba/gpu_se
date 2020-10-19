import numpy
import tqdm
import matplotlib.pyplot as plt
import sim_base
import itertools
import warnings
import matplotlib
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
from decorators import PickleJar


@PickleJar.pickle(path='bioreactor')
def step_test(percent, dt):
    """Does a simulation with given input changes and returns the outputs vs time

    Parameters
    ----------
    percent : numpy.array
        A (2,) array containing the percentage changes for the two inputs

    dt : the simulation period

    Returns
    -------
    ts, ys : numpy.array
        The times and outputs of the simulation
    """
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
    """Plot outputs vs time for various runs
    """
    percents = numpy.array([0.5, 0.7, 0.8, 1, 1.2, 1.3, 1.5])
    dts = [0.1]

    # u = numpy.array([0.06, 0.2])
    matplotlib.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(1, 2,
                             sharey='row',
                             figsize=(6.25 * 2, 5),
                             gridspec_kw={'wspace': 0.08}
                             )

    max_slope = 0
    argmax = 0

    for p1, p2, dt in itertools.product(percents, percents, dts):
        percent = numpy.array([p1, p2])
        ts, ys = step_test(percent, dt)
        # u_i = u * percent

        # Find the maximum slope
        Cg = ys[:, 0]
        Cga = numpy.abs(Cg - Cg[0])
        Cga_max = numpy.max(Cga)
        ts_max = ts[numpy.where(Cga == Cga_max)][0]
        slope = Cga_max / ts_max
        argmax = argmax
        if slope > max_slope:
            max_slope = slope
            Cg_max = Cg[numpy.where(ts == ts_max)]
            argmax = ts_max, Cg_max

        ax = axes[0]
        ax.plot(ts, ys[:, 2])
        ax.set_title(r'$C_{FA}$')
        ax.set_ylabel(r'$\frac{mmol}{L}$')
        ax.set_xlabel(r't ($min$)')
        ax.set_xlim(xmin=0, xmax=100)

        ax = axes[1]
        ax.plot(ts, ys[:, 0])
        ax.set_title(r'$C_{G}$')
        # ax.set_ylabel(r'$\frac{mmol}{L}$')
        ax.set_xlabel(r't ($min$)')
        ax.set_xlim(xmin=0, xmax=300)

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

        # plt.suptitle('Step tests')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('step_tests.pdf')
    plt.show()

    print(max_slope, argmax)


if __name__ == '__main__':
    plot()
