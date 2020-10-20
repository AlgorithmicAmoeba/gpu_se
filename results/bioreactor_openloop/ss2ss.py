import numpy
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import sim_base
import model


def plot_ss2ss():
    """Plot the transition from one steady state to another
    """
    # Simulation set-up
    end_time = 1000
    ts = numpy.linspace(0, end_time, end_time*10)
    dt = ts[1]

    bioreactor = model.Bioreactor(
        #                Ng,         Nx,      Nfa, Ne, Nh
        X0=numpy.array([3000 / 180, 1 / 24.6, 0 / 116, 0., 0.]),
        high_N=True
    )

    select_inputs = [0, 1]  # Fg_in, Fm_in
    select_outputs = [0, 2]  # Cg, Cfa

    state_pdf, measurement_pdf = sim_base.get_noise()

    # Initial values
    us = [numpy.array([0., 0.])]
    xs = [bioreactor.X.copy()]
    ys = [bioreactor.outputs(us[-1])]
    ys_meas = [bioreactor.outputs(us[-1])]

    not_cleared = True
    for t in tqdm.tqdm(ts[1:]):
        if t < 25:
            us.append(numpy.array([0., 0.]))
        elif t < 400:
            if not_cleared:
                bioreactor.X[[0, 2, 3, 4]] = 0
                not_cleared = False
                bioreactor.high_N = False

            us.append(numpy.array([0.06, 0.2]))
        elif t < 1000:
            us.append(numpy.array([0.04, 0.1]))
        else:
            us.append(us[-1])

        bioreactor.step(dt, us[-1])
        bioreactor.X += state_pdf.draw().get().squeeze()
        outputs = bioreactor.outputs(us[-1])
        ys.append(outputs.copy())
        outputs[select_outputs] += measurement_pdf.draw().get().squeeze()
        ys_meas.append(outputs)
        xs.append(bioreactor.X.copy())

    ys_meas = numpy.array(ys_meas)
    us = numpy.array(us)

    def add_time_lines(axis):
        for time in [400]:
            axis.axvline(time, color='black', alpha=0.4)
        axis.set_xlim([25, ts[-1]])

    matplotlib.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(1, 2,
                             figsize=(6.25 * 2, 5),
                             gridspec_kw={'wspace': 0.25}
                             )
    ax = axes[0]
    ax.plot(ts, us[:, select_inputs[1]], 'k')
    ax.plot(ts, us[:, select_inputs[0]], 'k--')

    ax.set_title(r'Inputs')
    ax.set_ylabel(r'$\frac{L}{min}$')
    ax.set_xlabel(r't ($min$)')
    ax.set_xlim([0, ts[-1]])
    ax.legend([r'$F_{m, in}$', r'$F_{G, in}$'])

    add_time_lines(ax)

    ax = axes[1]
    ax.plot(ts, ys_meas[:, 2], 'k')
    ax.plot(ts, ys_meas[:, 0], 'grey')
    ax.plot(ts, ys_meas[:, 3], 'k--')

    ax.set_ylim(ymax=1500)
    ax.set_title(r'Outputs')
    ax.set_ylabel(r'$\frac{mg}{L}$')
    ax.set_xlabel(r't ($min$)')
    ax.legend([r'$C_{FA}$', r'$C_{G}$', r'$C_{E}$'])

    add_time_lines(ax)

    # plt.subplot(2, 3, 6)
    # plt.plot(ts, ys[:, 1])
    # plt.title(r'$C_{X}$')
    # add_time_lines()

    # plt.suptitle('Openloop transition between steady states')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('ss2ss.pdf')
    plt.show()


if __name__ == '__main__':
    plot_ss2ss()
