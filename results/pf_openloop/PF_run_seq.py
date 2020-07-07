import numpy
import time
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import sim_base
import joblib


def prediction_run_seqs(N_part, N_runs, gpu):
    memory = joblib.Memory('cache/predict')
    # memory.clear()

    # noinspection PyShadowingNames
    @memory.cache
    def predict(N_particle, N_runs, gpu):
        times = []

        _, _, _, p = sim_base.get_parts(
            N_particles=N_particle,
            gpu=gpu
        )

        for _ in tqdm.tqdm(range(N_runs)):
            u, _ = sim_base.get_random_io()
            t = time.time()
            p.predict(u, 1.)
            times.append(time.time() - t)

        return numpy.array(times)

    N_particles = 2**numpy.arange(1, N_part, 0.5)
    run_seqs = numpy.array(
        [predict(int(N_particle), N_runs, gpu) for N_particle in tqdm.tqdm(N_particles)]
    )

    return N_particles, run_seqs


def update_run_seqs(N_part, N_runs, gpu):
    memory = joblib.Memory('cache/update')
    # memory.clear()

    # noinspection PyShadowingNames
    @memory.cache
    def update(N_particle, N_runs, gpu):
        times = []

        _, _, _, p = sim_base.get_parts(
            N_particles=N_particle,
            gpu=gpu
        )

        for j in range(N_runs):
            u, y = sim_base.get_random_io()
            t = time.time()
            p.update(u, y)
            times.append(time.time() - t)

        return numpy.array(times)

    N_particles = 2**numpy.arange(1, N_part, 0.5)
    run_seqs = numpy.array(
        [update(int(N_particle), N_runs, gpu) for N_particle in tqdm.tqdm(N_particles)]
    )

    return N_particles, run_seqs


def resample_run_seqs(N_part, N_runs, gpu):
    memory = joblib.Memory('cache/resample')
    # memory.clear()

    # noinspection PyShadowingNames
    @memory.cache
    def resample(N_particle, N_runs, gpu):
        times = []

        _, _, _, p = sim_base.get_parts(
            N_particles=N_particle,
            gpu=gpu
        )

        for j in range(N_runs):
            p.weights = numpy.random.random(size=p.N_particles)
            p.weights /= numpy.sum(p.weights)
            t = time.time()
            p.resample()
            times.append(time.time() - t)

        return numpy.array(times)

    N_particles = 2**numpy.arange(1, N_part, 0.5)
    run_seqs = numpy.array(
        [resample(int(N_particle), N_runs, gpu) for N_particle in tqdm.tqdm(N_particles)]
    )

    return N_particles, run_seqs


def get_run_seqs():
    """Returns the run sequences for all the runs

    Returns
    -------
    run_seqss : List
        [CPU; GPU] x [predict; update; resample] x [N_particles; run_seq]
    """
    run_seqss = [
        [
            prediction_run_seqs(20, 20, False),
            update_run_seqs(20, 100, False),
            resample_run_seqs(20, 100, False)
        ],
        [
            prediction_run_seqs(24, 100, True),
            update_run_seqs(24, 100, True),
            resample_run_seqs(24, 100, True)
        ]
    ]
    return run_seqss


def plot_example_benchmark():
    run_seqss = get_run_seqs()
    N_particles, run_seqs = run_seqss[0][1]
    run_seq = run_seqs[-1]

    print(numpy.log2(N_particles[-1]))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.semilogy(run_seq, 'kx')
    plt.title('Run sequence')
    plt.xlabel('Iterations')
    plt.ylabel('Time (s)')

    plt.subplot(1, 3, 2)
    plt.plot(run_seq[:-1], run_seq[1:], 'kx')
    plt.title('Lag chart')
    plt.xlabel(r'$X_{i-1}$')
    plt.ylabel(r'$X_{i}$')

    plt.subplot(1, 3, 3)
    plt.acorr(run_seq - numpy.average(run_seq))
    plt.title('Autocorrelation graph')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

    plt.suptitle(r'Benchmarking for CPU update with $N_p = 2^{19.5}$')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('benchmark.pdf')
    plt.show()


def plot_max_auto():
    run_seqss = get_run_seqs()

    for row in range(2):
        for col in range(3):
            plt.subplot(2, 3, row + 2*col + 1)
            N_parts, run_seqs = run_seqss[row][col]
            N_logs = numpy.log2(N_parts)

            for N_log, run_seq in zip(N_logs, run_seqs):
                x = run_seq - numpy.average(run_seq)
                Nx = len(x)
                maxlags = 1
                abs_cors = numpy.abs(numpy.correlate(x, x, mode="full"))
                abs_cors /= numpy.dot(x, x)
                abs_cors = abs_cors[Nx - 1 - maxlags:Nx + maxlags]
                abs_cors.sort()
                # if abs_cors[-2] > 0.9:
                #     plt.figure()
                #     pandas.plotting.autocorrelation_plot(run_seq)
                #     plt.show()
                #     plt.acorr(run_seq, maxlags=90)
                #     plt.show()
                #     plt.plot(run_seq[:-1], run_seq[1:], 'kx')
                #     plt.show()
                #     plt.semilogy(run_seq, 'kx')
                #     plt.show()
                #     return
                plt.plot(N_log, abs_cors[-2], 'kx')
            plt.ylim(0, 1)
    plt.show()


# noinspection PyUnresolvedReferences
def plot_run_seqs():
    run_seqss = get_run_seqs()

    cmap = matplotlib.cm.get_cmap('Spectral')
    norm = matplotlib.colors.Normalize(vmin=1, vmax=24)

    for row in range(2):
        for col in range(3):
            plt.subplot(2, 3, row + 2*col + 1)
            N_parts, run_seqs = run_seqss[row][col]
            N_logs = numpy.log2(N_parts)

            for N_log, run_seq in zip(N_logs, run_seqs):
                normailised_value = (N_log - N_logs[0]) / N_logs[-1]
                plt.semilogy(run_seq, '.', color=cmap(normailised_value))

    plt.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=norm,
            cmap=cmap
        ),
        ax=plt.gcf().get_axes()
    )

    plt.savefig('run_seqs.pdf')
    plt.show()


def plot_speed_up():
    run_seqss = get_run_seqs()

    for method in range(3):
        cpu_time = numpy.min(run_seqss[0][method][1], axis=1)
        gpu_time = numpy.min(run_seqss[1][method][1], axis=1)

        speed_up = cpu_time / gpu_time[:cpu_time.shape[0]]
        logN_part = numpy.log2(run_seqss[0][method][0])
        plt.semilogy(logN_part, speed_up, '.')

    plt.legend(['Predict', 'Update', 'Resample'])
    plt.show()


def plot_times():
    run_seqss = get_run_seqs()

    for device in range(2):
        plt.subplot(1, 2, device+1)
        for method in range(3):
            times = numpy.min(run_seqss[device][method][1], axis=1)
            logN_part = numpy.log2(run_seqss[device][method][0])
            plt.semilogy(logN_part, times, '.')

        plt.legend(['Predict', 'Update', 'Resample'])
    plt.show()


if __name__ == '__main__':
    plot_max_auto()
    # plot_example_benchmark()
    # plot_run_seqs()
    # plot_speed_up()
    # plot_times()
