import numpy
import time
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import sim_base
import joblib
import cupy
import statsmodels.tsa.stattools as stats_tools
import torch
import torch.utils.dlpack as torch_dlpack


class RunSequences:
    def __init__(self, function, path='cache/'):
        self.memory = joblib.Memory(path + function.__name__)
        self.function = self.memory.cache(function)

    def __call__(self, N_particles, N_runs, *args, **kwargs):
        run_seqs = numpy.array(
            [self.function(int(N_particle), N_runs, *args, **kwargs) for N_particle in N_particles]
        )

        return N_particles, run_seqs

    def clear(self, *args):
        self.function.call_and_shelve(*args).clear()

    @staticmethod
    def vectorize(function):
        return RunSequences(function)


@RunSequences.vectorize
def predict_run_seq(N_particle, N_runs, gpu):
    times = []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu
    )

    for _ in range(N_runs):
        u, _ = sim_base.get_random_io()
        t = time.time()
        p.predict(u, 1.)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
def update_run_seq(N_particle, N_runs, gpu):
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


@RunSequences.vectorize
def resample_run_seq(N_particle, N_runs, gpu):
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


@RunSequences.vectorize
def f_vectorize_run_seq(N_particle, N_runs, mem_gpu):
    times = []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True
    )

    for _ in tqdm.tqdm(range(N_runs)):
        u, _ = sim_base.get_random_io()
        if mem_gpu:
            u = cupy.asarray(u)
        t = time.time()
        p.f_vectorize(p.particles_device, u, 1.)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
def g_vectorize_run_seq(N_particle, N_runs, mem_gpu):
    times = []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True
    )

    for _ in tqdm.tqdm(range(N_runs)):
        u, _ = sim_base.get_random_io()
        if mem_gpu:
            u = cupy.asarray(u)
        t = time.time()
        # noinspection PyProtectedMember
        p.g_vectorize(p.particles_device, u, p._y_dummy)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
def state_pdf_draw_run_seq(N_particle, N_runs):
    times = []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True
    )

    for _ in tqdm.tqdm(range(N_runs)):
        t = time.time()
        p.state_pdf.draw(p.N_particles)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
def measurement_pdf_run_seq(N_particle, N_runs):
    times = []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True
    )

    for _ in tqdm.tqdm(range(N_runs)):
        es = p.measurement_pdf.draw(p.N_particles)
        t = time.time()
        p.measurement_pdf.pdf(es)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
def cumsum_run_seq(N_particle, N_runs):
    times = []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True
    )

    for _ in tqdm.tqdm(range(N_runs)):
        p.weights_device = cupy.random.uniform(size=p.N_particles)
        t = time.time()
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(p.weights_device).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]
        times.append(time.time() - t)

    return numpy.array(times)


def get_run_seqs():
    """Returns the run sequences for all the runs

    Returns
    -------
    run_seqss : List
        [CPU; GPU] x [predict; update; resample] x [N_particles; run_seq]
    """
    N_particles_cpu = 2**numpy.arange(1, 20, 0.5)
    N_particles_gpu = 2**numpy.arange(1, 24, 0.5)
    run_seqss = [
        [
            predict_run_seq(N_particles_cpu, 20, False),
            update_run_seq(N_particles_cpu, 100, False),
            resample_run_seq(N_particles_cpu, 100, False)
        ],
        [
            predict_run_seq(N_particles_gpu, 100, True),
            update_run_seq(N_particles_gpu, 100, True),
            resample_run_seq(N_particles_gpu, 100, True)
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
            plt.subplot(2, 3, 3*row + col + 1)
            N_parts, run_seqs = run_seqss[row][col]
            N_logs = numpy.log2(N_parts)

            for N_log, run_seq in zip(N_logs, run_seqs):
                abs_cors = numpy.abs(stats_tools.pacf(run_seq, nlags=10)[1:])
                plt.plot(N_log, numpy.max(abs_cors), 'kx')
            plt.ylim(0, 1)
            plt.xlim(0, 20)
            plt.axhline(0.2, color='r')
            plt.xlabel(r'$\log_2(N_p)$')

            if row == 1:
                plt.title(['Predict', 'Update', 'Resample'][col])
                if col == 0:
                    plt.ylabel('CPU', rotation=0)

            if col == 0 and row == 1:
                plt.ylabel('GPU', rotation=0)
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
