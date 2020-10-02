import numpy
import tqdm
import matplotlib.pyplot as plt
import sim_base
import matplotlib
from decorators import PickleJar


@PickleJar.pickle(path='bioreactor/perf_vs_cp/raw')
def get_simulation_performance(dt_control, monte_carlo):
    """Does a simulation with a given control period and returns the performance

    Parameters
    ----------
    dt_control : float
        Control period

    monte_carlo : int
        Index of the monte carlo run

    Returns
    -------
    performance : float
        ISE performance of the run

    """
    _ = monte_carlo
    end_time = 50
    ts = numpy.linspace(0, end_time, end_time*20)
    dt = ts[1]
    assert dt <= dt_control

    bioreactor, lin_model, K, _ = sim_base.get_parts(dt_control=dt_control)
    state_pdf, measurement_pdf = sim_base.get_noise()

    # Initial values
    us = [numpy.array([0.06, 0.2])]
    xs = [bioreactor.X.copy()]
    ys = [bioreactor.outputs(us[-1])]

    biass = []

    t_next = 0
    for t in ts[1:]:
        if t > t_next:
            # noinspection PyUnresolvedReferences
            U_temp = us[-1].copy()
            if K.y_predicted is not None:
                biass.append(lin_model.yn2d(ys[-1]) - K.y_predicted)

            # noinspection PyBroadException
            try:
                u = K.step(
                    lin_model.xn2d(xs[-1]),
                    lin_model.un2d(us[-1]),
                    lin_model.yn2d(ys[-1])
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
        outputs[lin_model.outputs] += measurement_pdf.draw().get().squeeze()
        ys.append(outputs.copy())
        xs.append(bioreactor.X.copy())

    ys = numpy.array(ys)

    return sim_base.performance(ys[:, lin_model.outputs], lin_model.yd2n(K.ysp), ts)


@PickleJar.pickle(path='bioreactor/perf_vs_cp/processed')
def generate_results():
    """Collects individual simulation results for performance runs

    Returns
    -------
    dt_controls, performances : list
        A list of control periods and performances
    """
    monte_carlos = 5
    dt_controls, performances = [], []
    for dt_control in tqdm.tqdm(numpy.linspace(0.1, 30, 20)):
        for monte_carlo in range(monte_carlos):
            y = get_simulation_performance(dt_control, monte_carlo)
            if y > 1e8:
                continue
            dt_controls.append(dt_control)
            performances.append(y)

    return dt_controls, performances


def plot_results():
    """Plots performances vs control periods
    """
    matplotlib.rcParams.update({'font.size': 9})
    plt.figure(figsize=(6.25/1.4, 5/1.4))

    dt_controls, performances = generate_results()
    plt.plot(dt_controls, performances, 'k.')
    plt.axvline(0.1, color='red')

    # plt.title("Closedloop performance vs control period")
    plt.ylabel(r'$P_{\mathrm{ISE}}$')
    plt.xlabel('Control period (min)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.axvline(1)
    plt.savefig('performance_vs_control_period.pdf')
    plt.show()


def plot_pretty_results():
    """Plots performances vs control periods.
    For use in a presentation.
    """
    plt.style.use('seaborn-deep')

    black = '#2B2B2D'
    # red = '#E90039'
    # orange = '#FF1800'
    white = '#FFFFFF'
    yellow = '#FF9900'

    plt.figure(figsize=(12.8, 9.6))
    plt.rcParams.update({'font.size': 16, 'text.color': white, 'axes.labelcolor': white,
                         'axes.edgecolor': white, 'xtick.color': white, 'ytick.color': white})

    plt.gcf().set_facecolor(black)
    plt.gca().set_facecolor(black)

    dt_controls, performances = generate_results()
    plt.plot(dt_controls, performances, '.', color=yellow)
    # plt.title("Closedloop performance vs control period")
    plt.ylabel(r'$P_{\mathrm{ITAE}}$')
    plt.xlabel('Control period (min)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('performance_vs_control_period.png', transparent=True)
    plt.show()


if __name__ == '__main__':
    plot_results()
    # plot_pretty_results()
