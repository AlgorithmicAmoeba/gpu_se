import numpy
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import sim_base


def simulate():
    """Performs no noise simulation"""
    # Simulation set-up
    end_time = 50
    ts = numpy.linspace(0, end_time, end_time*10)
    dt = ts[1]
    dt_control = 1
    assert dt <= dt_control

    bioreactor, lin_model, K, _ = sim_base.get_parts(dt_control=dt_control)

    # Initial values
    us = [numpy.array([0.06, 0.2])]
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
    biass = numpy.array(biass)

    print('Performance: ', sim_base.performance(ys[:, lin_model.outputs], lin_model.yd2n(K.ysp), ts))

    return ts, ys, lin_model, K, us, dt_control, biass, end_time


def plot():
    """Plots outputs, inputs and biases vs time
    for a closed loop simulation from a steady state to a set point
    """
    ts, ys, lin_model, K, us, dt_control, biass, end_time = simulate()

    matplotlib.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.25 * 3, 5),
        gridspec_kw={'wspace': 0.3}
    )

    ax = axes[0]
    ax.plot(ts, us[:, lin_model.inputs[1]], 'k')
    ax.plot(ts, us[:, lin_model.inputs[0]], 'k--')

    ax.set_title(r'Inputs')
    ax.set_ylabel(r'$\frac{L}{min}$')
    ax.set_xlabel(r't ($min$)')
    ax.legend([r'$F_{m, in}$', r'$F_{G, in}$'])
    ax.set_xlim([0, ts[-1]])

    ax = axes[1]
    ax.plot(ts, ys[:, 2], 'k')
    ax.plot(ts, ys[:, 0], 'grey')
    ax.plot(ts, ys[:, 3], 'k--')

    ax.set_title(r'Outputs')
    ax.set_ylabel(r'$\frac{mg}{L}$')
    ax.set_xlabel(r't ($min$)')
    ax.set_xlim([0, ts[-1]])
    ax.legend([r'$C_{FA}$', r'$C_{G}$', r'$C_{E}$'])

    ax.axhline(lin_model.yd2n(K.ysp)[1], color='red')
    ax.axhline(lin_model.yd2n(K.ysp)[0], color='red', linestyle='--')

    ax = axes[2]
    ax.plot(
        numpy.arange(dt_control, end_time, dt_control),
        biass[:, 1],
        'k'
    )
    ax.plot(
        numpy.arange(dt_control, end_time, dt_control),
        biass[:, 0],
        'k--'
    )
    ax.legend([r'$C_{FA}$', r'$C_G$'])
    ax.set_title('bias')
    ax.set_ylabel(r'$\frac{mg}{L}$')
    ax.set_xlabel(r't ($min$)')
    ax.set_xlim([0, ts[-1]])

    # plt.suptitle('Closedloop bioreactor without noise')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('no_noise.pdf', bbox_inches='tight')
    plt.show()


def plot_pretty():
    """Plots outputs, inputs and biases vs time
    for a closed loop simulation without noise from a steady state to a set point.
    For use in a presentation.
    """

    ts, ys, lin_model, K, us, dt_control, biass, end_time = simulate()
    plt.style.use('seaborn-deep')

    black = '#2B2B2D'
    red = '#E90039'
    orange = '#FF1800'
    white = '#FFFFFF'
    yellow = '#FF9900'

    plt.figure(figsize=(12.8, 9.6))
    plt.rcParams.update({'font.size': 16, 'text.color': white, 'axes.labelcolor': white,
                         'axes.edgecolor': white, 'xtick.color': white, 'ytick.color': white})

    plt.gcf().set_facecolor(black)

    plt.subplot(2, 3, 1)
    plt.plot(ts, ys[:, 2], color=orange)
    plt.axhline(lin_model.yd2n(K.ysp)[1], color=white)
    plt.title(r'$C_{FA}$')
    plt.xlim([0, ts[-1]])
    plt.gca().set_facecolor(black)

    plt.subplot(2, 3, 2)
    plt.plot(ts, ys[:, 0], color=orange)
    plt.axhline(lin_model.yd2n(K.ysp)[0], color=white)
    plt.title(r'$C_{G}$')
    plt.xlim([0, ts[-1]])
    plt.gca().set_facecolor(black)

    plt.subplot(2, 3, 3)
    plt.plot(ts, ys[:, 3], color=orange)
    plt.title(r'$C_{E}$')
    plt.xlim([0, ts[-1]])
    plt.gca().set_facecolor(black)

    plt.subplot(2, 3, 4)
    plt.plot(ts, us[:, lin_model.inputs[1]], color=red)
    plt.title(r'$F_{m, in}$')
    plt.xlim([0, ts[-1]])
    plt.gca().set_facecolor(black)

    plt.subplot(2, 3, 5)
    plt.plot(ts, us[:, lin_model.inputs[0]], color=red)
    plt.title(r'$F_{G, in}$')
    plt.xlim([0, ts[-1]])
    plt.gca().set_facecolor(black)

    plt.subplot(2, 3, 6)
    plt.plot(
        numpy.arange(dt_control, end_time, dt_control),
        biass[:, 1],
        color=red
    )
    plt.plot(
        numpy.arange(dt_control, end_time, dt_control),
        biass[:, 0],
        color=yellow
    )
    plt.legend([r'$C_{FA}$', r'$C_G$'], facecolor=black)
    plt.title('bias')
    plt.xlim([0, ts[-1]])
    plt.gca().set_facecolor(black)

    # plt.suptitle('Closedloop bioreactor without noise')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('no_noise_pretty.png', transparent=True)
    plt.show()


if __name__ == '__main__':
    plot()
    # plot_pretty()
