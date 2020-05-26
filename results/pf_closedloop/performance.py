import numpy
import tqdm
import matplotlib.pyplot as plt
import model.LinearModel
import sim_base
import pandas


def get_simulation_performance(N_particles, dt_control=1):
    # Simulation set-up
    end_time = 50
    ts = numpy.linspace(0, end_time, end_time*10)
    dt = ts[1]
    assert dt <= dt_control

    bioreactor, lin_model, K, pf = sim_base.get_parts(
        dt_control=dt_control,
        N_particles=N_particles,

    )
    state_pdf, measurement_pdf = sim_base.get_noise()

    # Initial values
    us = [numpy.array([0.06, 5/180, 0.2])]
    xs = [bioreactor.X.copy()]
    ys = [bioreactor.outputs(us[-1])]
    ys_meas = [bioreactor.outputs(us[-1])]
    ys_pf = [
        model.Bioreactor.static_outputs(
                (pf.weights_device @ pf.particles_device).get(),
                us[-1]
            )
    ]

    biass = []

    t_next = 0
    for t in ts[1:]:
        if t > t_next:
            U_temp = us[-1].copy()
            if K.y_predicted is not None:
                biass.append(lin_model.yn2d(ys_meas[-1]) - K.y_predicted)

            pf.update(us[-1], ys_meas[-1][lin_model.outputs])
            pf.resample()
            x_pf = (pf.weights_device @ pf.particles_device).get()
            u = K.step(lin_model.xn2d(x_pf), lin_model.un2d(us[-1]), lin_model.yn2d(ys_meas[-1]))
            U_temp[lin_model.inputs] = lin_model.ud2n(u)
            us.append(U_temp.copy())
            t_next += dt_control
        else:
            us.append(us[-1])

        bioreactor.step(dt, us[-1])
        bioreactor.X += state_pdf.draw().get()
        outputs = bioreactor.outputs(us[-1])
        ys.append(outputs.copy())
        outputs[lin_model.outputs] += measurement_pdf.draw().get()
        ys_meas.append(outputs)
        xs.append(bioreactor.X.copy())

        pf.predict(us[-1], dt)
        ys_pf.append(
            numpy.array(
                model.Bioreactor.static_outputs(
                    (pf.weights_device @ pf.particles_device).get(),
                    us[-1]
                )
            )
        )

    ys_meas = numpy.array(ys_meas)

    return sim_base.performance(ys_meas[:, lin_model.outputs], lin_model.yd2n(K.ysp), ts)


def generate_results(redo=False, low=1, high=20, repeat=1):
    try:
        if redo:
            raise FileNotFoundError

        df = pandas.read_csv('performance.csv', index_col=0)
    except FileNotFoundError:
        df = pandas.DataFrame(columns=['log2_Ns', 'performance'])

    for rep in range(repeat):
        log2_Ns = numpy.arange(low, high)
        performances = numpy.array([
            get_simulation_performance(2**log2_N) for log2_N in tqdm.tqdm(log2_Ns)
        ])

        res = numpy.vstack([log2_Ns, performances])
        df_new = pandas.DataFrame(res.T, columns=['log2_Ns', 'performance'])
        df = df.append(df_new).sort_values(by='log2_Ns')
        df.to_csv('performance.csv')


def plot_results():
    df = pandas.read_csv('performance.csv')
    plt.plot(df['log2_Ns'], df['performance'], '.')
    plt.title("Closedloop performance vs number of particles")
    plt.ylabel(r'$P_{\mathrm{ITAE}}$')
    plt.xlabel(r'$\log_2(N)$ particles')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('performance.pdf')
    plt.show()


if __name__ == '__main__':
    generate_results(redo=False, repeat=5)
    plot_results()
