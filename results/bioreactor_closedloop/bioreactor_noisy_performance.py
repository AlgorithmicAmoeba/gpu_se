import numpy
import tqdm
import matplotlib.pyplot as plt
import pandas
import sim_base


def get_simulation_performance(dt_control):
    end_time = 50
    ts = numpy.linspace(0, end_time, end_time*10)
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


def generate_results(redo=False, number=50, low=0.1, high=30.):
    try:
        if redo:
            raise FileNotFoundError

        df = pandas.read_csv('bioreactor_noisy_performance.csv', index_col=0)
    except FileNotFoundError:
        df = pandas.DataFrame(columns=['dt_controls', 'performance'])

    dt_controls = numpy.round(
        numpy.random.uniform(
            low=low,
            high=high,
            size=number
        ),
        1
    )
    #
    performances = numpy.array([
        get_simulation_performance(dt_control) for dt_control in tqdm.tqdm(dt_controls)
    ])

    res = numpy.vstack([dt_controls, performances])
    df_new = pandas.DataFrame(res.T, columns=['dt_controls', 'performance'])
    df = df.append(df_new).sort_values(by='dt_controls')
    df.to_csv('bioreactor_noisy_performance.csv')


def plot_results():
    df = pandas.read_csv('bioreactor_noisy_performance.csv')
    plt.plot(df['dt_controls'], df['performance'], 'k.')
    plt.axvline(0.1, color='red')

    # plt.title("Closedloop performance vs control period")
    plt.ylabel(r'$P_{\mathrm{ISE}}$')
    plt.xlabel('Control period (min)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.axvline(1)
    plt.savefig('noisy_performance.pdf')
    plt.show()


def plot_pretty_results():
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

    df = pandas.read_csv('bioreactor_noisy_performance.csv')
    plt.plot(df['dt_controls'], df['performance'], '.', color=yellow)
    # plt.title("Closedloop performance vs control period")
    plt.ylabel(r'$P_{\mathrm{ITAE}}$')
    plt.xlabel('Control period (min)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('noisy_performance_pretty.png', transparent=True)
    plt.show()


if __name__ == '__main__':
    # generate_results(redo=False, number=100)
    plot_results()
    # plot_pretty_results()
