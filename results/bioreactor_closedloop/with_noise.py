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

    state_pdf, measurement_pdf = sim_base.get_noise()

    # Initial values
    us = [numpy.array([0.06, 0.2])]
    xs = [bioreactor.X.copy()]
    ys = [bioreactor.outputs(us[-1])]
    ys_meas = [bioreactor.outputs(us[-1])]

    biass = []

    t_next = 0
    for t in tqdm.tqdm(ts[1:]):
        if t > t_next:
            U_temp = us[-1].copy()
            if K.y_predicted is not None:
                biass.append(lin_model.yn2d(ys_meas[-1]) - K.y_predicted)

            # noinspection PyBroadException
            try:
                u = K.step(
                    lin_model.xn2d(xs[-1]),
                    lin_model.un2d(us[-1]),
                    lin_model.yn2d(ys_meas[-1])
                )
            except:
                u = numpy.array([0.04, 0.1])
            U_temp[lin_model.inputs] = lin_model.ud2n(u)
            us.append(numpy.maximum(0, U_temp.copy()))
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

    ys = numpy.array(ys)
    ys_meas = numpy.array(ys_meas)
    us = numpy.array(us)
    biass = numpy.array(biass)

    print('Performance: ', sim_base.performance(ys[:, lin_model.outputs], lin_model.yd2n(K.ysp), ts))

    return ts, ys, ys_meas, lin_model, K, us, dt_control, biass, end_time


def plot():
    """Plots outputs, inputs and biases vs time
    for a closed loop simulation with noise from a steady state to a set point.
    """
    ts, ys, ys_meas, lin_model, K, us, dt_control, biass, end_time = simulate()

    matplotlib.rcParams.update({'font.size': 20})
    plt.figure(figsize=(6.25 * 3, 5 * 2))

    plt.subplot(2, 3, 1)
    plt.plot(ts, ys_meas[:, 2], 'k')
    plt.axhline(lin_model.yd2n(K.ysp)[1], color='red')
    plt.title(r'$C_{FA}$')
    plt.ylabel(r'$\frac{mmol}{L}$')
    plt.xlabel(r't ($min$)')
    plt.xlim([0, ts[-1]])

    plt.subplot(2, 3, 2)
    plt.plot(ts, ys_meas[:, 0], 'k')
    plt.axhline(lin_model.yd2n(K.ysp)[0], color='red')
    plt.title(r'$C_{G}$')
    plt.ylabel(r'$\frac{mmol}{L}$')
    plt.xlabel(r't ($min$)')
    plt.xlim([0, ts[-1]])

    plt.subplot(2, 3, 3)
    plt.plot(ts, ys[:, 3], 'k')
    plt.title(r'$C_{E}$')
    plt.ylabel(r'$\frac{mmol}{L}$')
    plt.xlabel(r't ($min$)')
    plt.xlim([0, ts[-1]])

    plt.subplot(2, 3, 4)
    plt.plot(ts, us[:, lin_model.inputs[1]], 'k')
    plt.title(r'$F_{m, in}$')
    plt.ylabel(r'$\frac{L}{min}$')
    plt.xlabel(r't ($min$)')
    plt.xlim([0, ts[-1]])

    plt.subplot(2, 3, 5)
    plt.plot(ts, us[:, lin_model.inputs[0]], 'k')
    plt.title(r'$F_{G, in}$')
    plt.ylabel(r'$\frac{L}{min}$')
    plt.xlabel(r't ($min$)')
    plt.xlim([0, ts[-1]])

    plt.subplot(2, 3, 6)
    plt.plot(
        numpy.arange(dt_control, end_time, dt_control),
        biass[:, 1],
        'k'
    )
    plt.plot(
        numpy.arange(dt_control, end_time, dt_control),
        biass[:, 0],
        'k--'
    )
    plt.legend([r'$C_{FA}$', r'$C_G$'])
    plt.title('bias')
    plt.ylabel(r'$\frac{mmol}{L}$')
    plt.xlabel(r't ($min$)')
    plt.xlim([0, ts[-1]])

    # plt.suptitle('Closedloop bioreactor with noise')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('with_noise.pdf')
    plt.show()


def plot_pretty():
    """Plots outputs, inputs and biases vs time
    for a closed loop simulation with noise from a steady state to a set point.
    For use in a presentation.
    """
    ts, ys, ys_meas, lin_model, K, us, dt_control, biass, end_time = simulate()

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
    plt.plot(ts, ys_meas[:, 2], color=orange)
    plt.plot(ts, ys[:, 2], color=yellow)
    plt.axhline(lin_model.yd2n(K.ysp)[1], color=white)
    plt.legend(['measured', 'true'], facecolor=black)
    plt.title(r'$C_{FA}$')
    plt.xlim([0, ts[-1]])
    plt.gca().set_facecolor(black)

    plt.subplot(2, 3, 2)
    plt.plot(ts, ys_meas[:, 0], color=orange)
    plt.plot(ts, ys[:, 0], color=yellow)
    plt.axhline(lin_model.yd2n(K.ysp)[0], color=white)
    plt.legend(['measured', 'true'], facecolor=black)
    plt.title(r'$C_{G}$')
    plt.xlim([0, ts[-1]])
    plt.gca().set_facecolor(black)

    plt.subplot(2, 3, 3)
    plt.plot(ts, ys_meas[:, 3], color=orange)
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
    plt.legend([r'$C_G$', r'$C_{FA}$'], facecolor=black)
    plt.title('bias')
    plt.gca().set_facecolor(black)

    # plt.suptitle('Closedloop bioreactor with noise')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('with_noise_pretty.png', transparent=True)
    plt.show()


if __name__ == '__main__':
    plot()
    # plot_pretty()
