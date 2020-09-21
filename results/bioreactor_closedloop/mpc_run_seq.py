import numpy
import time
import joblib
import tqdm
import sim_base
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stats_tools

memory = joblib.Memory('cache/mpc')


@memory.cache
def mpc_run_seq(N_runs):
    """Performs a run sequence on the MPC step function for a
     number of runs

    Parameters
    ----------
    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """

    times = []

    bioreactor, lin_model, K, _ = sim_base.get_parts(dt_control=0.1)

    state_pdf, measurement_pdf = sim_base.get_noise()

    # Initial values
    dt = 0.1
    us = [numpy.array([0.06, 0.2])]
    xs = [bioreactor.X.copy()]
    ys = [bioreactor.outputs(us[-1])]
    ys_meas = [bioreactor.outputs(us[-1])]

    biass = []

    for _ in tqdm.tqdm(range(N_runs)):
        U_temp = us[-1].copy()
        if K.y_predicted is not None:
            biass.append(lin_model.yn2d(ys_meas[-1]) - K.y_predicted)

        t = time.time()
        # noinspection PyBroadException
        try:
            u = K.step(
                lin_model.xn2d(xs[-1]),
                lin_model.un2d(us[-1]),
                lin_model.yn2d(ys_meas[-1])
            )
        except:
            u = numpy.array([0.06, 0.2])
        times.append(time.time() - t)

        U_temp[lin_model.inputs] = lin_model.ud2n(u)
        us.append(U_temp.copy())

        bioreactor.step(dt, us[-1])
        bioreactor.X += state_pdf.draw().get().squeeze()
        outputs = bioreactor.outputs(us[-1])
        ys.append(outputs.copy())
        outputs[lin_model.outputs] += measurement_pdf.draw().get().squeeze()
        ys_meas.append(outputs)
        xs.append(bioreactor.X.copy())

    return numpy.array(times)


def plot_benchmarks():
    run_seq = mpc_run_seq(1000)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(run_seq, 'kx')
    plt.title('Run sequence')
    plt.xlabel('Iterations')
    plt.ylabel('Time (s)')

    plt.subplot(1, 3, 2)
    plt.plot(run_seq[:-1], run_seq[1:], 'kx')
    plt.title('Lag chart')
    plt.xlabel(r'$X_{i-1}$')
    plt.ylabel(r'$X_{i}$')

    plt.subplot(1, 3, 3)
    abs_cors = numpy.abs(stats_tools.pacf(run_seq, nlags=10)[1:])
    plt.plot(abs_cors, 'kx')
    plt.title('Autocorrelation graph')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

    # plt.suptitle(r'Benchmarking for MPC')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('mpc_benchmark.pdf')
    plt.show()


if __name__ == '__main__':
    plot_benchmarks()
    print(numpy.median(mpc_run_seq(1000)))
