import numpy
import time
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import sim_base
import joblib


def prediction_run_seqs(N_part, N_runs, gpu):
    memory = joblib.Memory('cache/predict')

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


# noinspection PyUnresolvedReferences
def plot_run_seqs():
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

            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))

    plt.show()


def plot_speed_up():
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
    for method in range(3):
        cpu_time = numpy.min(run_seqss[0][method][1], axis=1)
        gpu_time = numpy.min(run_seqss[1][method][1], axis=1)

        speed_up = cpu_time / gpu_time[:cpu_time.shape[0]]
        logN_part = numpy.log2(run_seqss[0][method][0])
        plt.semilogy(logN_part, speed_up, '.')

    plt.legend(['Predict', 'Update', 'Resample'])
    plt.show()


def plot_times():
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
    for device in range(2):
        plt.subplot(1, 2, device+1)
        for method in range(3):
            times = numpy.min(run_seqss[device][method][1], axis=1)
            logN_part = numpy.log2(run_seqss[device][method][0])
            plt.semilogy(logN_part, times, '.')

        plt.legend(['Predict', 'Update', 'Resample'])
    plt.show()


if __name__ == '__main__':
    # plot_run_seqs()
    plot_speed_up()
    plot_times()
