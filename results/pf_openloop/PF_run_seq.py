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
import filter.particle


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
        p.f_vectorize(p.particles, u, 1.)
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
        p.g_vectorize(p.particles, u, p._y_dummy)
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
        p.weights = cupy.random.uniform(size=p.N_particles)

        t = time.time()
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(p.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def parallel_resample_run_seq(N_particle, N_runs):
    times = []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True
    )

    for _ in tqdm.tqdm(range(N_runs)):
        p.weights = cupy.random.uniform(size=p.N_particles)
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(p.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]

        t = time.time()
        sample_index = cupy.zeros(p.N_particles, dtype=cupy.int64)
        random_number = cupy.float64(cupy.random.rand())

        filter.particle.ParallelParticleFilter._parallel_resample[p._bpg, p._tpb](
            cumsum, sample_index,
            random_number,
            p.N_particles
        )
        times.append(time.time() - t)

    return numpy.array(times)


# noinspection PyProtectedMember
@RunSequences.vectorize
def index_copying_run_seq(N_particle, N_runs):
    times = []

    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True
    )

    for _ in tqdm.tqdm(range(N_runs)):
        p.weights = cupy.random.uniform(size=p.N_particles)
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(p.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]

        sample_index = cupy.zeros(p.N_particles, dtype=cupy.int64)
        random_number = cupy.float64(cupy.random.rand())

        filter.particle.ParallelParticleFilter._parallel_resample[p._bpg, p._tpb](
            cumsum, sample_index,
            random_number,
            p.N_particles
        )

        t = time.time()
        p.particles = cupy.asarray(p.particles)[sample_index]
        p.weights = cupy.full(p.N_particles, 1 / p.N_particles)
        times.append(time.time() - t)

    return numpy.array(times)


@RunSequences.vectorize
def no_op_run_seq(N_time, N_runs):
    times = []

    for _ in tqdm.tqdm(range(N_runs)):
        t = time.time()
        time.sleep(N_time)
        times.append(time.time() - t)

    return numpy.array(times)


def cpu_gpu_run_seqs():
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


def pf_sub_routine_run_seqs():
    N_particles = 2**numpy.arange(1, 24, 0.5)
    f_vec_gpu_mem = lambda x, y: f_vectorize_run_seq(x, y, True)
    f_vec_cpu_mem = lambda x, y: f_vectorize_run_seq(x, y, False)
    g_vec_gpu_mem = lambda x, y: g_vectorize_run_seq(x, y, True)
    g_vec_cpu_mem = lambda x, y: g_vectorize_run_seq(x, y, False)

    funcs = [
        f_vec_gpu_mem, f_vec_cpu_mem, state_pdf_draw_run_seq,
        g_vec_gpu_mem, g_vec_cpu_mem, measurement_pdf_run_seq,
        cumsum_run_seq, parallel_resample_run_seq, index_copying_run_seq
             ]

    run_seqs = [func(N_particles, 50) for func in funcs]

    return run_seqs


def plot_example_benchmark():
    N_times = [10]  # numpy.logspace(1e-3, 10, 30)
    N_times, run_seqs = no_op_run_seq(N_times, 100)
    run_seq = run_seqs[-1]

    print(N_times[-1])

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
    abs_cors = numpy.abs(stats_tools.pacf(run_seq, nlags=10)[1:])
    plt.plot(abs_cors, 'kx')
    plt.title('Autocorrelation graph')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

    plt.suptitle(r'Benchmarking for CPU update with $N_p = 2^{19.5}$')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('benchmark.pdf')
    plt.show()


def plot_max_auto():
    run_seqss = cpu_gpu_run_seqs()

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

            if row == 0:
                plt.title(['Predict', 'Update', 'Resample'][col])

            if row == 0 and col == 0:
                plt.ylabel('CPU', rotation=0)

            if col == 0 and row == 1:
                plt.ylabel('GPU', rotation=0)
    plt.show()


# noinspection PyUnresolvedReferences
def plot_run_seqs():
    run_seqss = cpu_gpu_run_seqs()

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
    run_seqss = cpu_gpu_run_seqs()

    for method in range(3):
        cpu_time = numpy.max(run_seqss[0][method][1], axis=1)
        gpu_time = numpy.max(run_seqss[1][method][1], axis=1)

        speed_up = cpu_time / gpu_time[:cpu_time.shape[0]]
        logN_part = numpy.log2(run_seqss[0][method][0])
        plt.semilogy(logN_part, speed_up, '.')

    plt.legend(['Predict', 'Update', 'Resample'])
    plt.show()


def plot_times():
    run_seqss = cpu_gpu_run_seqs()

    for device in range(2):
        plt.subplot(1, 2, device+1)
        for method in range(3):
            times = numpy.max(run_seqss[device][method][1], axis=1)
            logN_part = numpy.log2(run_seqss[device][method][0])
            plt.semilogy(logN_part, times, '.')

        plt.legend(['Predict', 'Update', 'Resample'])
    plt.show()


def plot_sub_routine_max_auto():
    run_seqss = pf_sub_routine_run_seqs()
    names = [
        'f (GPU memory)', 'f (CPU memory)', 'State noise - draw',
        'g (GPU memory)', 'g (CPU memory)', 'Measurement noise - pdf',
        'cumsum', 'Nicely algorithm', 'Index copying'
     ]

    for i, (N_parts, run_seqs) in enumerate(run_seqss):
        plt.subplot(3, 3, i+1)
        N_logs = numpy.log2(N_parts)

        for N_log, run_seq in zip(N_logs, run_seqs):
            abs_cors = numpy.abs(stats_tools.pacf(run_seq, nlags=10)[1:])
            plt.plot(N_log, numpy.max(abs_cors), 'kx')
        plt.ylim(0, 1)
        plt.xlim(0, 20)
        plt.axhline(0.2, color='r')
        plt.xlabel(r'$\log_2(N_p)$')
        plt.title(names[i])
    plt.show()


def plot_sub_routine_times():
    names = [
        'f', 'f (memory copy)', 'State noise - draw',
        'g', 'g ((memory copy)', 'Measurement noise - pdf',
        'cumsum', 'Nicely algorithm', 'Index copying'
    ]

    run_seqss = pf_sub_routine_run_seqs()

    run_seqss[1] = (run_seqss[1][0], run_seqss[1][1] - run_seqss[0][1])
    run_seqss[4] = (run_seqss[4][0], run_seqss[4][1] - run_seqss[3][1])

    for i, (N_parts, run_seqs) in enumerate(run_seqss):
        plt.subplot(3, 3, i + 1)
        times = numpy.max(run_seqs, axis=1)
        logN_part = numpy.log2(N_parts)
        plt.semilogy(logN_part, times, '.')
        plt.title(names[i])

    plt.show()


def plot_sub_routine_fractions():
    names = [
        'f', 'f (memory copy)', 'State noise - draw',
        'g', 'g (memory copy)', 'Measurement noise - pdf',
        'cumsum', 'Nicely algorithm', 'Index copying'
    ]
    func_seqss = pf_sub_routine_run_seqs()

    func_seqss[1] = (func_seqss[1][0], func_seqss[1][1] - func_seqss[0][1])
    func_seqss[4] = (func_seqss[4][0], func_seqss[4][1] - func_seqss[3][1])

    plt.figure(figsize=(15, 5))
    for i, func_indxs in enumerate([[0, 1, 2], [3, 4, 5], [6, 7, 8]]):
        plt.subplot(1, 3, i+1)
        N_parts = func_seqss[0][0]
        logN_part = numpy.log2(N_parts)

        total_times = numpy.zeros_like(N_parts)
        for func_indx in func_indxs:
            total_times += numpy.average(abs(func_seqss[func_indx][1]), axis=1)

        bottom = None
        for func_indx in func_indxs:
            times = numpy.average(abs(func_seqss[func_indx][1]), axis=1)
            frac_times = times / total_times
            plt.bar(logN_part, frac_times, width=0.55, bottom=bottom, label=names[func_indx])
            if bottom is None:
                bottom = frac_times
            else:
                bottom += frac_times
        plt.legend()
        plt.title(['Predict', 'Update', 'Resample'][i])
        if i == 0:
            plt.ylabel('Fraction of runtime')
        plt.xlabel(r'$\log_2(N_p)$')

    plt.tight_layout()
    plt.savefig('pf_frac_breakdown.pdf')
    plt.show()


if __name__ == '__main__':
    plot_sub_routine_max_auto()
    plot_sub_routine_fractions()
    plot_example_benchmark()

    # pf_sub_routine_run_seqs()
    # plot_max_auto()
    # plot_times()
    # plot_speed_up()
